# ML Interview Guide — single-repo build for all four volumes.
ENGINE ?= tectonic
VOLS := deep-learning nlp search-recommendation conventional-ml

.PHONY: all check sync-style $(VOLS)

all: check $(VOLS)

$(VOLS):
	cd volumes/$@ && $(ENGINE) main.tex

check:
	python3 ci/check_drift.py --root volumes

sync-style:
	@for v in $(VOLS); do \
	  grep -q "usepackage{essentials}" volumes/$$v/main.tex \
	    && cp style/essentials.sty volumes/$$v/essentials.sty && echo "synced sty -> $$v" \
	    || true; \
	  cp style/program_map.tex volumes/$$v/program_map.tex && echo "synced map -> $$v" || true; \
	  for k in part0_kernel core50 numbers_card; do \
	    cp kernel/$$k.tex volumes/$$v/$$k.tex || true; \
	  done; echo "synced kernel -> $$v"; \
	done

.PHONY: index
index:
	python3 tools/gen_question_index.py

.PHONY: verify
verify: check
	python3 tools/gen_question_index.py --check
