# eeg-slm — common commands

PY := python
PIP := pip

.PHONY: help setup setup-dev clean download explore forward profile train lint test info

help:
	@echo "Available targets:"
	@echo "  setup       - Install runtime dependencies"
	@echo "  setup-dev   - Install runtime + dev dependencies + editable install"
	@echo "  download    - Download initial EEG dataset (PhysioNet EEGMMIDB)"
	@echo "  explore     - Run the first-look exploration script"
	@echo "  forward     - Run EEGLeJEPA forward + backward on real EEG (Session 2)"
	@echo "  profile     - Profile forward/backward timing on the active device"
	@echo "  train       - Pretrain EEGLeJEPA on EEGMMIDB (Session 3, default 300 steps)"
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
