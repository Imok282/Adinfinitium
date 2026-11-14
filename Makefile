.PHONY: install test check demo-tile bench bench-cpu diagrams

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

diagrams:           ## re-render docs/diagrams from .dot and .mmd sources
	bash docs/diagrams/render.sh
