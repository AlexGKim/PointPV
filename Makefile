PY := $(shell conda info --base)/envs/generic/bin/python

.PHONY: test test-fast test-slow test-all validate benchmark

# Fast: excludes flip (requires FLIP/CAMB) and slow (scaling tests)
test-fast:
	$(PY) -m pytest tests/ -v -m "not flip and not slow"

# Medium: includes slow scaling tests, excludes flip
test-slow:
	$(PY) -m pytest tests/ -v -m "not flip"

# Full: all tests (requires FLIP/CAMB installed)
test-all:
	$(PY) -m pytest tests/ -v

# Science validation: fsigma8 recovery with both methods (N=200, ~10 s)
validate:
	$(PY) scripts/validate_fsigma8.py --n 200 --n-grid 40

# Runtime scaling benchmark (N=100, 1000; ~30 s on laptop)
benchmark:
	$(PY) scripts/benchmark_scaling.py --sizes 100 1000 --n-repeats 3
