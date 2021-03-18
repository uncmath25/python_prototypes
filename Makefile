.PHONY: clean build flake run

SHELL := /bin/bash
IMAGE_NAME := uncmath25/jupyter-notebook
CONTAINER_NAME := jupyter_notebook
IMAGE_HOME_DIR := /home/jovyan

default: run

clean:
	@echo "*** Cleaning the repo ***"
	rm -rf output/*
	find . -name '.ipynb_checkpoints' -type d | xargs rm -rf

build: clean
	@echo "*** Building the docker images ***"
	docker build -t $(IMAGE_NAME) .

flake: build
	@echo "*** Linting the python scripts ***"
	docker run \
		--rm \
		-p 8888:8888 \
		-v "$$(pwd)/scripts:$(IMAGE_HOME_DIR)/scripts" \
		$(IMAGE_NAME) bash -c "flake8 --ignore='E501' scripts"

run: build
	@echo "*** Running Jupyter notebook virtual environment ***"
	mkdir -p .jupyter .local
	docker run \
		--rm \
		--name $(CONTAINER_NAME) \
		-p 8888:8888 \
		-e JUPYTER_ENABLE_LAB=yes \
		-v "$$(pwd)/.jupyter:$(IMAGE_HOME_DIR)/.jupyter" \
		-v "$$(pwd)/.local:$(IMAGE_HOME_DIR)/.local" \
		-v "$$(pwd)/notebooks:$(IMAGE_HOME_DIR)/notebooks" \
		$(IMAGE_NAME)
