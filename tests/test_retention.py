"""Unit tests for M3 retention constraint checks and Holm-Bonferroni adjustment."""

from __future__ import annotations

from src.evaluation.retention import PERPLEXITY_PASSAGES, RETENTION_PROMPTS, holm_bonferroni


def test_retention_suite_structure() -> None:
    assert len(RETENTION_PROMPTS) == 30
    ids = [t.task_id for t in RETENTION_PROMPTS]
    assert len(set(ids)) == 30
    by_cat: dict[str, int] = {}
    for t in RETENTION_PROMPTS:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    assert len(by_cat) == 6
    assert all(cnt == 5 for cnt in by_cat.values())
    assert len(PERPLEXITY_PASSAGES) == 10
    assert all(len(p.split()) >= 35 for p in PERPLEXITY_PASSAGES)


def test_retention_verifiers_accept_valid_and_reject_invalid() -> None:
    valid_samples = {
        "ret_bullet_01": "- Hiking\n- Kayaking\n- Cycling",
        "ret_bullet_02": "- Whisk\n- Spatula\n- Ladle\n- Tongs",
        "ret_bullet_03": "- Piano\n- Violin\n- Flute\n- Cello\n- Oboe",
        "ret_bullet_04": "- Wear warm wool gloves\n- Check road frost early\n- Carry dry extra socks",
        "ret_bullet_05": "- Pack water\n- Watch clouds\n- Bring a headlamp\n- Tell someone your route",
        "ret_json_01": '{"city": "Kyoto", "country": "Japan", "continent": "Asia"}',
        "ret_json_02": '{"item": "bicycle", "wheels": 2, "material": "aluminum"}',
        "ret_json_03": '{"title": "Dune", "author": "Frank Herbert", "genre": "Science fiction"}',
        "ret_json_04": '{"planet": "Mars", "color": "rust red", "region": "Tharsis"}',
        "ret_json_05": '{"drink": "green tea", "origin": "Japan", "temperature": "hot"}',
        "ret_forbid_01": "Cool misty rain falls softly on dark moss.",
        "ret_forbid_02": "Silent white pine trees rustle softly in deep frost.",
        "ret_forbid_03": "Crisp sea breeze fills white sails at bright dawn.",
        "ret_forbid_04": "Warm golden loaves make every home smell sweet and cozy.",
        "ret_forbid_05": "Clear twilight stars shine calmly above dark pines.",
        "ret_words_01": "Golden leaves drift gently across the quiet morning park path.",
        "ret_words_02": "A warm mug of tea makes quiet evening reading feel especially peaceful.",
        "ret_words_03": "Hot coffee tastes great while watching distant towns.",
        "ret_words_04": "Deep nightly sleep restores steady focus, sharp memory, balanced mood, and daily energy cleanly.",
        "ret_words_05": "Fishing boats rest quietly inside the glassy morning harbor.",
        "ret_affix_01": "NOTE: Compost enriches garden soil with organic matter. It also helps retain steady moisture.",
        "ret_affix_02": "Public libraries offer free books and quiet study spaces for everyone. That completes the note.",
        "ret_affix_03": "SUMMARY: Staying hydrated supports alertness and clear concentration across long workdays.",
        "ret_affix_04": "Keep one offline copy of your critical folders every week. End of reminder.",
        "ret_affix_05": "BRIEFING: Ocean tides rise and fall regularly as lunar gravity pulls coastal waters.",
        "ret_case_01": "BRIGHT STARS GLOW CLEARLY OVER SILENT DESERT DUNES ALL NIGHT LONG.",
        "ret_case_02": "clear mountain rivers wind steadily through mossy cedar valleys after spring rain.",
        "ret_case_03": "STEADY PATIENCE HELPS SOLVE DIFFICULT WORK WITHOUT RUSHING OR PANIC.",
        "ret_case_04": "dark storm clouds gather swiftly over open wheat fields before dusk.",
        "ret_case_05": "HISTORICAL MAPS SHOW HOW TRADE ROUTES CONNECTED DISTANT COASTAL PORTS.",
    }
    bad_samples = {
        "ret_bullet_01": "Here are three:\n- Hiking\n- Kayaking\n- Cycling",
        "ret_json_01": '{"city": "Kyoto", "country": "Japan"}',
        "ret_forbid_01": "Gentle rain falls silently everywhere.",
        "ret_words_01": "Too short.",
        "ret_affix_01": "Compost enriches garden soil steadily over several seasons.",
        "ret_case_01": "Bright stars glow clearly over silent desert dunes all night long.",
    }
    tasks_by_id = {t.task_id: t for t in RETENTION_PROMPTS}
    for tid, good in valid_samples.items():
        ok, msg = tasks_by_id[tid].check(good)
        assert ok, f"Expected valid sample to pass {tid}: {msg}"
    for tid, bad in bad_samples.items():
        ok, _ = tasks_by_id[tid].check(bad)
        assert not ok, f"Expected invalid sample to fail {tid}"


def test_holm_bonferroni_adjustment() -> None:
    res = holm_bonferroni({"c1": 0.01, "c2": 0.04, "c3": 0.03, "c4": 0.80})
    assert abs(res["c1"] - 0.04) < 1e-12
    assert abs(res["c3"] - 0.09) < 1e-12
    assert abs(res["c2"] - 0.09) < 1e-12  # monotonicity: max(0.09, 2*0.04=0.08) == 0.09
    assert abs(res["c4"] - 0.80) < 1e-12
    capped = holm_bonferroni({"a": 0.70, "b": 0.80})
    assert capped == {"a": 1.0, "b": 1.0}
