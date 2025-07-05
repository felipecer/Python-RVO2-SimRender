# Ruta absoluta al directorio actual (Python-RVO2-SimRender)
PROJECT_ROOT := $(shell pwd)

# Ruta completa al entorno virtual
VENV_DIR := $(PROJECT_ROOT)/venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip

# Repositorio externo
PYRVO2_REPO := $(PROJECT_ROOT)/../PYRVO2-RL
PYRVO2_GIT := https://github.com/felipecer/PYRVO2-RL.git

all: setup

# Clona PYRVO2-RL si no existe
$(PYRVO2_REPO):
	git clone $(PYRVO2_GIT) $(PYRVO2_REPO)

# Crea entorno virtual
$(VENV_DIR):
	python3.10 -m venv $(VENV_DIR)

# Instala dependencias del proyecto
deps: $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Instala PYRVO2-RL desde fuera, pero usando el python del entorno virtual
install-pyrvo2: $(PYRVO2_REPO) deps
	cd $(PYRVO2_REPO) && $(PYTHON) -m pip install .

# Instalación completa
setup: install-pyrvo2

# Ejecuta el entorno con run.py
run-circle:
	PYTHONPATH=$(PROJECT_ROOT) $(PYTHON) run.py rl_environments/single_agent/miac/v2/circle.py

run-script:
	$(PYTHON) run.py $(PROJECT_ROOT)/$(script)

# Limpia entorno virtual y repositorio clonado
clean:
	rm -rf $(VENV_DIR)
	rm -rf $(PYRVO2_REPO)
