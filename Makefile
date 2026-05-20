# eeg-slm — common commands

PY := python
PIP := pip

.PHONY: help setup setup-dev clean download explore forward profile train probe lint test info

help:
	@echo "Available targets:"
	@echo "  setup       - Install runtime dependencies"
	@echo "  setup-dev   - Install runtime + dev dependencies + editable install"
	@echo "  download    - Download initial EEG dataset (PhysioNet EEGMMIDB)"
	@echo "  explore     - Run the first-look exploration script"
	@echo "  forward     - Run EEGLeJEPA forward + backward on real EEG (Session 2)"
	@echo "  profile     - Profile forward/backward timing on the active device"
	@echo "  train       - Pretrain EEGLeJEPA on EEGMMIDB (Session 3, default 300 steps)"
	@echo "  probe       - Run linear probe on a pretrained checkpoint vs random init"
	@echo "  lint        - Run ruff"
	@echo "  test        - Run pytest"
	@echo "  info        - Show environment info"
	@echo "  clean       - Remove caches and build artifacts"

setup:
	$(PIP) install -e .

setup-dev:
	$(PIP) install -e ".[dev,training]"

download:
	$(PY) scripts/01_download_data.py

explore:
	$(PY) scripts/02_explore_data.py

forward:
	$(PY) scripts/03_model_forward.py

profile:
	$(PY) scripts/03b_profile_forward.py

train:
	$(PY) scripts/04_train.py

probe:
	$(PY) scripts/05_linear_probe.py --ckpt runs/lambda-1.0/model_final.pt --subjects 1 2 3

probe-rest:
	$(PY) scripts/05_linear_probe.py --ckpt runs/lambda-1.0/model_final.pt --subjects 1 2 3 --task rest_vs_activity

probe-bci-baseline:
	$(PY) scripts/07_probe_bci_iv_2a.py

download-20:
	$(PY) scripts/01b_download_range.py --start 4 --end 20

lint:
	ruff check src tests scripts

test:
	pytest

info:
	$(PY) -c "import sys, platform, torch, mne; print(f'Python: {sys.version.split()[0]}'); print(f'Platform: {platform.platform()}'); print(f'PyTorch: {torch.__version__}'); print(f'MNE: {mne.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'CUDA available: {torch.cuda.is_available()}')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info
