"""Bounded HTTP Range client for remote Safetensors inspection without full downloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch


class FullBodyFallbackError(RuntimeError):
    """Raised when a remote server returns HTTP 200 full-body instead of 206 Partial Content."""


class ByteBudgetExceededError(RuntimeError):
    """Raised when a remote fetch would exceed the configured byte budget."""


_DTYPE_MAP: dict[str, tuple[int, Any]] = {
    "BF16": (2, torch.bfloat16),
    "F16": (2, torch.float16),
    "F32": (4, torch.float32),
    "F64": (8, torch.float64),
    "I8": (1, torch.int8),
    "U8": (1, torch.uint8),
    "I16": (2, torch.int16),
    "I32": (4, torch.int32),
    "I64": (8, torch.int64),
    "BOOL": (1, torch.bool),
}


_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


def parse_content_range(header_val: str | None) -> tuple[int, int, int | None]:
    """Validate and parse an HTTP Content-Range header."""
    if not header_val:
        raise FullBodyFallbackError("Missing Content-Range header on partial response")
    m = _CONTENT_RANGE_RE.match(header_val.strip())
    if not m:
        raise ValueError(f"Malformed Content-Range header: {header_val!r}")
    start, end = int(m.group(1)), int(m.group(2))
    total = None if m.group(3) == "*" else int(m.group(3))
    return start, end, total


def decode_safetensors_bytes(raw: bytes, dtype_str: str, shape: list[int]) -> torch.Tensor:
    """Decode raw little-endian Safetensors bytes into a CPU torch.Tensor."""
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported Safetensors dtype: {dtype_str}")
    elem_size, torch_dtype = _DTYPE_MAP[dtype_str]
    expected_numel = 1
    for d in shape:
        expected_numel *= int(d)
    expected_bytes = expected_numel * elem_size
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Byte length mismatch for dtype={dtype_str}, shape={shape}: got {len(raw)}, expected {expected_bytes}"
        )
    # Create writable copy of raw bytes into uint8 tensor, then view as target dtype
    buf = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(buf).view(torch_dtype).reshape(shape)


class RemoteSafetensorsClient:
    """Strict HTTP Range client for pinned Hugging Face Safetensors files."""

    def __init__(
        self,
        repo_id: str,
        revision: str,
        filename: str = "model.safetensors",
        cache_dir: str | Path | None = None,
        byte_budget: int = 50_000_000,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.byte_budget = int(byte_budget)
        self.max_retries = int(max_retries)
        self.timeout_seconds = float(timeout_seconds)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        self.bytes_read = 0
        self.cache_hits = 0
        self.request_logs: list[dict[str, Any]] = []
        self._header_cache: dict[str, Any] | None = None
        self._total_file_bytes: int | None = None
        self._transport = transport

    def _build_headers(self, start: int, end: int) -> dict[str, str]:
        headers = {"Range": f"bytes={start}-{end}"}
        token = os.environ.get("HF_TOKEN")
        if token:
            # Never log headers/tokens; only attach in memory when present.
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _cache_file(self, start: int, end: int) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_repo = self.repo_id.replace("/", "__")
        safe_file = self.filename.replace("/", "__")
        return self.cache_dir / f"{safe_repo}__{self.revision[:12]}__{safe_file}__{start}_{end}.bin"

    def fetch_range(self, start: int, end: int) -> bytes:
        """Fetch inclusive byte range [start, end], rejecting non-206 responses before downloading body."""
        if start < 0 or end < start:
            raise ValueError(f"Invalid byte range [{start}, {end}]")
        expected_len = end - start + 1

        cache_path = self._cache_file(start, end)
        if cache_path is not None and cache_path.exists():
            data = cache_path.read_bytes()
            if len(data) == expected_len:
                self.cache_hits += 1
                self.request_logs.append(
                    {
                        "range": [start, end],
                        "nbytes": len(data),
                        "cache_hit": True,
                        "http_status": None,
                        "content_range": None,
                        "latency_ms": 0.0,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "partial_hash_only": True,
                    }
                )
                return data

        if self.bytes_read + expected_len > self.byte_budget:
            raise ByteBudgetExceededError(
                f"Fetching {expected_len} bytes would exceed byte_budget={self.byte_budget} "
                f"(already read {self.bytes_read})"
            )

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                client_kwargs: dict[str, Any] = {
                    "follow_redirects": True,
                    "timeout": self.timeout_seconds,
                }
                if self._transport is not None:
                    client_kwargs["transport"] = self._transport

                with httpx.Client(**client_kwargs) as client:
                    with client.stream("GET", self.url, headers=self._build_headers(start, end)) as resp:
                        if resp.status_code == 200:
                            raise FullBodyFallbackError(
                                f"Server returned HTTP 200 (full-body fallback rejected) for {self.url}"
                            )
                        if resp.status_code != 206:
                            raise RuntimeError(
                                f"Expected HTTP 206 Partial Content, got {resp.status_code} for {self.url}"
                            )
                        cr_raw = resp.headers.get("Content-Range")
                        cr_start, cr_end, cr_total = parse_content_range(cr_raw)
                        if cr_start != start or cr_end != end:
                            raise RuntimeError(
                                f"Content-Range mismatch: expected {start}-{end}, got {cr_raw}"
                            )
                        if cr_total is not None:
                            self._total_file_bytes = cr_total

                        chunks: list[bytes] = []
                        received = 0
                        for chunk in resp.iter_bytes():
                            received += len(chunk)
                            if self.bytes_read + received > self.byte_budget:
                                raise ByteBudgetExceededError(
                                    f"Stream exceeded byte_budget={self.byte_budget}"
                                )
                            if received > expected_len + 4096:
                                raise RuntimeError(
                                    f"Server streamed {received} bytes for expected {expected_len}"
                                )
                            chunks.append(chunk)
                        data = b"".join(chunks)
                        if len(data) != expected_len:
                            raise RuntimeError(
                                f"Incomplete byte range: expected {expected_len}, got {len(data)}"
                            )

                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.bytes_read += len(data)
                if cache_path is not None:
                    cache_path.write_bytes(data)

                self.request_logs.append(
                    {
                        "range": [start, end],
                        "nbytes": len(data),
                        "cache_hit": False,
                        "http_status": 206,
                        "content_range": cr_raw,
                        "total_file_bytes": self._total_file_bytes,
                        "latency_ms": round(latency_ms, 2),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "partial_hash_only": True,
                    }
                )
                return data
            except (FullBodyFallbackError, ByteBudgetExceededError):
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Failed range fetch [{start}, {end}] after {self.max_retries} attempts: {last_err}")

    def fetch_header(self) -> dict[str, Any]:
        """Fetch and parse only the Safetensors JSON header."""
        if self._header_cache is not None:
            return self._header_cache

        len_bytes = self.fetch_range(0, 7)
        (header_len,) = struct.unpack("<Q", len_bytes)
        if header_len > 32_000_000:
            raise ValueError(f"Refusing unexpectedly huge Safetensors header: {header_len} bytes")
        header_bytes = self.fetch_range(8, 8 + header_len - 1)
        raw_header = json.loads(header_bytes.decode("utf-8"))
        self._header_cache = {
            "header_len_bytes": header_len,
            "data_start_offset": 8 + header_len,
            "total_file_bytes": self._total_file_bytes,
            "raw_header": raw_header,
        }
        return self._header_cache

    def fetch_tensor(self, tensor_name: str) -> torch.Tensor:
        """Fetch an entire named tensor via exact byte offsets."""
        hdr = self.fetch_header()
        info = hdr["raw_header"].get(tensor_name)
        if info is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in remote header")
        data_start = int(hdr["data_start_offset"])
        rel_start, rel_end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        abs_start = data_start + rel_start
        abs_end = data_start + rel_end - 1
        raw = self.fetch_range(abs_start, abs_end)
        return decode_safetensors_bytes(raw, info["dtype"], list(info["shape"]))

    def fetch_tensor_row_slice(
        self,
        tensor_name: str,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        """Fetch contiguous rows [row_start, row_end) of a row-major tensor via HTTP Range."""
        hdr = self.fetch_header()
        info = hdr["raw_header"].get(tensor_name)
        if info is None:
            raise KeyError(f"Tensor '{tensor_name}' not found in remote header")
        shape = list(info["shape"])
        if len(shape) < 1:
            raise ValueError(f"Cannot row-slice scalar tensor '{tensor_name}'")
        r0 = max(0, row_start)
        r1 = min(shape[0], row_end)
        if r0 >= r1:
            raise ValueError(f"Empty row slice [{r0}, {r1}) for shape {shape}")

        dtype_str = info["dtype"]
        elem_size, _ = _DTYPE_MAP[dtype_str]
        row_elems = 1
        for d in shape[1:]:
            row_elems *= int(d)
        row_bytes = row_elems * elem_size

        data_start = int(hdr["data_start_offset"])
        rel_start = int(info["data_offsets"][0])
        abs_start = data_start + rel_start + r0 * row_bytes
        abs_end = data_start + rel_start + r1 * row_bytes - 1
        raw = self.fetch_range(abs_start, abs_end)
        slice_shape = [r1 - r0] + shape[1:]
        return decode_safetensors_bytes(raw, dtype_str, slice_shape)

    def summary(self) -> dict[str, Any]:
        """Return network/cache accounting summary without leaking credentials."""
        total = self._total_file_bytes
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "byte_budget": self.byte_budget,
            "bytes_read": self.bytes_read,
            "cache_hits": self.cache_hits,
            "total_file_bytes": total,
            "byte_fraction_read": (self.bytes_read / total) if total else None,
            "requests": self.request_logs,
        }
