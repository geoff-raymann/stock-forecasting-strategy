# -----------------------------
# 📈 Stock Forecasting Makefile
# -----------------------------

# === ENVIRONMENT ===
PYTHON = python
PIP = pip

# === COMMON ===
VENV_ACTIVATE = source venv/bin/activate

# === TARGETS ===

# 🛠️ Setup virtual environment
venv:
	$(PYTHON) -m venv venv
	$(VENV_ACTIVATE) && $(PIP) install --upgrade pip

# 🔍 Install dependencies
install:
	$(VENV_ACTIVATE) && $(PIP) install -r requirements.txt

# ✅ Validate your local Python environment
check-env:
	$(VENV_ACTIVATE) && $(PYTHON) -c "import statsmodels, sklearn, prophet, yfinance; print('✅ Python env looks good!')"

# 📥 Pull live data
data:
	$(VENV_ACTIVATE) && $(PYTHON) src/data_loader.py

# 🔍 Validate fetched data
validate:
	$(VENV_ACTIVATE) && $(PYTHON) tools/validate_data.py

# 🔄 Run full batch pipeline
batch:
	$(VENV_ACTIVATE) && $(PYTHON) src/batch_runner.py

# 🐳 Build Docker image
docker-build:
	docker build -t stock-dashboard .

# 🐳 Run Docker container locally
docker-run:
	docker run -p 8501:8501 stock-dashboard

# 🐳 Compose up (dashboard + batch_worker if defined)
compose-up:
	docker-compose up --build

# 🧹 Stop containers & clean up
compose-down:
	docker-compose down

# 📦 Full clean install
setup:
	make venv
	make install
	make check-env

# 🏃 Shortcut: run local end-to-end test
run-all:
	make data
	make validate
	make batch
