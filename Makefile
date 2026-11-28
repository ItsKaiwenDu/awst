PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)

.PHONY: help test verify check figures overleaf

help:
	@echo "Available targets:"
	@echo "  make test      Run the pytest test suite"
	@echo "  make verify    Verify manuscript numbers, LaTeX syntax, figures, and citations"
	@echo "  make check     Run both test and verify"
	@echo "  make figures   Regenerate publication figures from results/*.json"
	@echo "  make overleaf  Rebuild awst_overleaf.zip from paper/"

test:
	$(PYTEST) -q

verify:
	$(PYTHON) src/paper/verify_paper_numbers.py

check: test verify

figures:
	@if $(PYTHON) -c "import matplotlib" 2>/dev/null; then \
		MPLCONFIGDIR=/tmp/mplconfig $(PYTHON) src/paper/generate_figures.py; \
	else \
		MPLCONFIGDIR=/tmp/mplconfig python3 src/paper/generate_figures.py; \
	fi

overleaf:
	rm -f awst_overleaf.zip
	cd paper && zip -q ../awst_overleaf.zip main.tex references.bib figures/fig1_split_trajectory.pdf figures/fig2_m3_forest_contrasts.pdf figures/fig3_edit_magnitude_vs_ood_ppl.pdf
