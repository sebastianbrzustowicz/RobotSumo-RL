VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

MODEL_DIR = models/history
PPO_MODELS = $(MODEL_DIR)/PPO/*.pt
A2C_MODELS = $(MODEL_DIR)/A2C/*.pt
MASTER_MODEL = models/sumo_push_master.pt

FAV_PPO = models/favourite/PPO
FAV_A2C = models/favourite/A2C

PPO_SCRIPT = src/agents/PPO/trainer.py
A2C_SCRIPT = src/agents/A2C/trainer.py
PPO_TEST_SCRIPT = src/agents/PPO/test_PPO.py
A2C_TEST_SCRIPT = src/agents/A2C/test_A2C.py
CROSS_PLAY_SCRIPT = src/common/cross_play.py
ELO_SCRIPT = src/common/elo_tournament.py

.PHONY: install clean-models train-ppo train-a2c train-ppo-cont train-a2c-cont \
        test-ppo test-a2c cross-play tournament remove-venv test lint

define pick_models
	mkdir -p $(2)
	rm -f $(2)/*.pt
	ls $(1)/model_v*.pt 2>/dev/null | sort -V -r | sed -n 'p;n' | head -n 5 | xargs -I {} cp {} $(2)/
endef

install:
	@echo "Creating virtual environment and installing all dependencies..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	$(PIP) install -e ".[dev]"
	@echo "----------------------------------------------------------"
	@echo "DONE. To activate venv in your shell, run: source $(VENV)/bin/activate"
	@echo "----------------------------------------------------------"

clean-models:
	rm -f $(PPO_MODELS) $(A2C_MODELS) $(MASTER_MODEL)

train-ppo:
	rm -f $(PPO_MODELS) $(MASTER_MODEL)
	$(PYTHON) $(PPO_SCRIPT)

train-a2c:
	rm -f $(A2C_MODELS) $(MASTER_MODEL)
	$(PYTHON) $(A2C_SCRIPT)

train-ppo-cont:
	$(PYTHON) $(PPO_SCRIPT)

train-a2c-cont:
	$(PYTHON) $(A2C_SCRIPT)

test-ppo:
	$(PYTHON) $(PPO_TEST_SCRIPT)

test-a2c:
	$(PYTHON) $(A2C_TEST_SCRIPT)

cross-play:
	$(PYTHON) $(CROSS_PLAY_SCRIPT)

tournament:
	@$(call pick_models,$(MODEL_DIR)/PPO,$(FAV_PPO))
	@$(call pick_models,$(MODEL_DIR)/A2C,$(FAV_A2C))
	$(PYTHON) $(ELO_SCRIPT)

tournament-manually:
	$(PYTHON) $(ELO_SCRIPT)

remove-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check --select I --fix .
	$(VENV)/bin/black .