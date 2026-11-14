.PHONY: install test check demo-tile bench bench-cpu swarm swarm-smoke diagrams

install:            ## laptop / CI install (CPU JAX)
	pip install -e ".[dev]"

test:               ## run all tests
	pytest

check:              ## validate config + print derived numbers
	python -m contracts.mission

demo-tile:          ## synthetic crater tile (GENERATED) so everyone can work before real LOLA tiles
	python -m data_tools.make_demo_tile

bench: demo-tile    ## 4090 check: 10,000 landers with classic guidance
	python -m scripts.bench_swarm --n 10000

bench-cpu:          ## quick laptop check
	python -m scripts.bench_swarm --n 500 --flat

# 4090 overnight run (KAN-105/100). Re-run the SAME command after a crash: --resume continues.
#   make swarm                                   # trains on tight_fuel exams
#   make swarm EXAMS=tight_fuel,nav_drift_near_hazard,combined_stress SWARM_FLAGS=--lookahead   # needs KAN-100 look-ahead merged
RUN ?= overnight1
EXAMS ?= tight_fuel
SWARM_FLAGS ?=
swarm: demo-tile    ## 4090 overnight run: 10,000 pilots x 300 generations, trained on exams, resumable
	python -m swarm.run_swarm --pop 10000 --generations 300 --run $(RUN) --exams $(EXAMS) \
		--exam-targets 3 --exam-missions 2 --resume $(SWARM_FLAGS)

swarm-smoke:        ## laptop smoke test of the swarm (needs the "swarm" extra)
	python -m swarm.run_swarm --pop 64 --generations 3 --missions 2 --eval-missions 64 --flat --run smoke

diagrams:           ## re-render docs/diagrams from .dot and .mmd sources
	bash docs/diagrams/render.sh
