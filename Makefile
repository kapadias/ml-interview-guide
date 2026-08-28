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
	  cp style/essentials.sty volumes/$$v/essentials.sty 2>/dev/null && echo "synced -> $$v" || true; \
	done
