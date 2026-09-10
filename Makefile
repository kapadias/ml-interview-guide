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

# The drill deck: the four volumes re-cut into spoken answers for breadth and
# depth rounds. questions.base.json is the input; everything else regenerates.
.PHONY: drill drill-pdf drill-web
drill: drill-pdf drill-web

drill/questions.json: drill/questions.base.json drill/slices/*.json \
                      drill/drops.json drill/drops_additions.json drill/copyedits.json
	python3 tools/merge_drill.py
	python3 tools/normalize_drill.py

drill-pdf: drill/questions.json
	python3 tools/render_drill_pdf.py
	cd drill && $(ENGINE) drill.tex

drill-web: drill/questions.json
	python3 tools/render_drill_web.py
