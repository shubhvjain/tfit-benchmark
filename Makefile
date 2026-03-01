# Environment 
mode ?= local
include .env.$(mode)
export

# Common 
SCRIPTS_DIR      = /app/scripts
CONTAINERS_DIR   = containers
WS_DATA          = /workspace/data
WS_INPUT         = /workspace/input
WS_TEMP          = /workspace/temp
WS_OUTPUT        = /workspace/output
WS_RESULT_INPUT  = /workspace/result_input
WS_RESULT_OUTPUT = /workspace/result_output

# Container runtime config
ifeq ($(container_app),docker)
CONTAINER_RUNTIME = docker run --rm -w /app
COREGTOR_IMAGE = tfit-coregtor:latest
COREGNET_IMAGE = tfit-coregnet:latest
# NETREM_IMAGE = tfit-netrem:latest
BIND_VOLUME = \
    -v $(PWD)/scripts:$(SCRIPTS_DIR) \
    -v $(DATA_PATH):$(WS_DATA) \
    -v $(EXP_INPUT_PATH):$(WS_INPUT) \
    -v $(EXP_TEMP_PATH):$(WS_TEMP) \
    -v $(EXP_OUTPUT_PATH):$(WS_OUTPUT) \
    -v $(ANALYSIS_INPUT_PATH):$(WS_RESULT_INPUT) \
    -v $(ANALYSIS_OUTPUT_PATH):$(WS_RESULT_OUTPUT)
CONTAINER_ENV = \
    -e DATA_PATH=$(WS_DATA) \
    -e EXP_INPUT_PATH=$(WS_INPUT) \
    -e EXP_TEMP_PATH=$(WS_TEMP) \
    -e EXP_OUTPUT_PATH=$(WS_OUTPUT) \
    -e MODE=$(mode) \
    -e ANALYSIS_INPUT_PATH=$(WS_RESULT_INPUT) \
    -e ANALYSIS_OUTPUT_PATH=$(WS_RESULT_OUTPUT)
BUILD_COREGTOR = docker build -f $(CONTAINERS_DIR)/coregtor.Dockerfile -t $(COREGTOR_IMAGE) .
BUILD_COREGNET = docker build -f $(CONTAINERS_DIR)/coregnet.Dockerfile -t $(COREGNET_IMAGE) .
else ifeq ($(container_app),apptainer)
CONTAINER_RUNTIME = apptainer exec
COREGTOR_IMAGE = $(CONTAINER_PATH)/tfit-coregtor.sif
COREGNET_IMAGE = $(CONTAINER_PATH)/tfit-coregnet.sif
BIND_VOLUME = \
    --bind $(PWD)/scripts:$(SCRIPTS_DIR) \
    --bind $(DATA_PATH):$(WS_DATA) \
    --bind $(EXP_INPUT_PATH):$(WS_INPUT) \
    --bind $(EXP_TEMP_PATH):$(WS_TEMP) \
    --bind $(EXP_OUTPUT_PATH):$(WS_OUTPUT) \
    --bind $(ANALYSIS_INPUT_PATH):$(WS_RESULT_INPUT) \
    --bind $(ANALYSIS_OUTPUT_PATH):$(WS_RESULT_OUTPUT)
CONTAINER_ENV = \
    --env DATA_PATH=$(WS_DATA),EXP_INPUT_PATH=$(WS_INPUT),EXP_TEMP_PATH=$(WS_TEMP),EXP_OUTPUT_PATH=$(WS_OUTPUT),MODE=$(mode),ANALYSIS_INPUT_PATH=$(WS_RESULT_INPUT),ANALYSIS_OUTPUT_PATH=$(WS_RESULT_OUTPUT)
BUILD_COREGTOR = apptainer build $(COREGTOR_IMAGE) $(CONTAINERS_DIR)/coregtor.def
BUILD_COREGNET = apptainer build $(COREGNET_IMAGE) $(CONTAINERS_DIR)/coregnet.def
else
$(error Unknown container_app: "$(container_app)". Use docker or apptainer)
endif

RUN = $(CONTAINER_RUNTIME) $(BIND_VOLUME) $(CONTAINER_ENV)

# Targets
.PHONY: build-coregtor build-coregnet containers datasets new-exp new-analysis analysis help

build-coregtor: ## Build coregtor container
    $(BUILD_COREGTOR)

build-coregnet: ## Build coregnet container
    $(BUILD_COREGNET)

containers: build-coregtor build-coregnet ## Build all containers

datasets: ## Download all datasets required
	poetry run python scripts/datasets.py --all

new-exp: ## Create new exp file. Pass name=exp1
	poetry run python scripts/util_project_files.py new exp $(name)

exp-init: ## Initialize an experiment
	poetry run python scripts/exp_init.py init $(name)

run-coregtor: ## Run coregtor pipeline. Actions: run, result, update_status, reset_failed, reset_claimed
	$(CONTAINER_RUNTIME) $(BIND_VOLUME) $(CONTAINER_ENV) $(COREGTOR_IMAGE) python $(SCRIPTS_DIR)/run_coregtor.py \
		$(action) $(id) $(dataset) $(if $(filter reset_claimed,$(action)),$(worker),) $(if $(filter run,$(action)),$(if $(worker),--worker $(worker),) $(if $(batch),--batch $(batch),),)


new-analysis: ## Create new analysis file. Pass name=analysis1
	poetry run python scripts/util_project_files.py new analysis $(name)

analysis: ## Run an analysis file. Pass id=<id>
	poetry run python scripts/analysis.py run $(id)

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/Makefile://' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
