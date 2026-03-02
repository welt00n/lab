# ──────────────────────────────────────────────────────────────
#  lab — Hamiltonian Physics Lab
# ──────────────────────────────────────────────────────────────

SHELL   := /bin/bash
PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
VENV    := .venv

# NVIDIA pip packages install libs under site-packages/nvidia/*/lib*.
# Export paths so numba's CUDA JIT compiler finds libcudart, libnvvm, etc.
SITE    := $(VENV)/lib/python3.12/site-packages
NV_LIBS := $(SITE)/nvidia/cuda_runtime/lib:$(SITE)/nvidia/cuda_nvcc/nvvm/lib64:$(SITE)/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH := $(NV_LIBS):$(LD_LIBRARY_PATH)
export CUDA_HOME       := $(SITE)/nvidia/cuda_nvcc

NH      ?= 40
NA      ?= 60
HMIN    ?= 0.1
HMAX    ?= 5.0
AXIS    ?= x

GRID    := --nh $(NH) --na $(NA) --hmin $(HMIN) --hmax $(HMAX) --axis $(AXIS)

.DEFAULT_GOAL := help

# ── Setup ────────────────────────────────────────────────────

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV)/bin/activate ## Install all dependencies into .venv
	$(PIP) install -r requirements.txt

# ── Experiments ──────────────────────────────────────────────

.PHONY: coin
coin: install ## Run coin drop (CPU batch)
	$(PYTHON) main.py coin $(GRID)

.PHONY: coin-video
coin-video: install ## Run coin drop (CPU batch) + save video
	$(PYTHON) main.py coin $(GRID) --save-video

.PHONY: coin-live
coin-live: install ## Run coin drop (live 3D dashboard)
	$(PYTHON) main.py coin $(GRID) --live

.PHONY: coin-gpu
coin-gpu: install ## Run coin drop (GPU CUDA)
	$(PYTHON) main.py coin $(GRID) --gpu

.PHONY: coin-gpu-video
coin-gpu-video: install ## Run coin drop (GPU CUDA) + save video
	$(PYTHON) main.py coin $(GRID) --gpu --save-video

.PHONY: cube
cube: install ## Run cube drop (CPU batch)
	$(PYTHON) main.py cube $(GRID)

.PHONY: cube-video
cube-video: install ## Run cube drop (CPU batch) + save video
	$(PYTHON) main.py cube $(GRID) --save-video

.PHONY: cube-live
cube-live: install ## Run cube drop (live 3D dashboard)
	$(PYTHON) main.py cube $(GRID) --live

.PHONY: cube-gpu
cube-gpu: install ## Run cube drop (GPU CUDA)
	$(PYTHON) main.py cube $(GRID) --gpu

.PHONY: cube-gpu-video
cube-gpu-video: install ## Run cube drop (GPU CUDA) + save video
	$(PYTHON) main.py cube $(GRID) --gpu --save-video

# ── Quick demos (small grid, fast) ──────────────────────────

.PHONY: demo-coin
demo-coin: install ## Quick coin demo (10×15 grid + video)
	$(PYTHON) main.py coin --nh 10 --na 15 --save-video

.PHONY: demo-cube
demo-cube: install ## Quick cube demo (10×15 grid + video)
	$(PYTHON) main.py cube --nh 10 --na 15 --save-video

# ── Testing ──────────────────────────────────────────────────

.PHONY: test
test: install ## Run full test suite
	$(PYTEST) tests/ -v

.PHONY: test-fast
test-fast: install ## Run tests (fail-fast, short output)
	$(PYTEST) tests/ -x -q

# ── Housekeeping ─────────────────────────────────────────────

.PHONY: clean
clean: ## Remove results, __pycache__, and .pyc files
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true
	rm -rf results/

.PHONY: clean-venv
clean-venv: ## Remove the virtual environment
	rm -rf $(VENV)

.PHONY: clean-all
clean-all: clean clean-venv ## Remove everything (results + venv)

# ── Help ─────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@printf '\n  \033[1m\033[36mlab — Hamiltonian Physics Lab\033[0m\n\n'
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[32m%-18s\033[0m %s\n", $$1, $$2}'
	@printf '\n  \033[2mOverride grid params: make coin NH=20 NA=30 HMIN=0.5 HMAX=3.0 AXIS=y\033[0m\n\n'
