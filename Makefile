.PHONY: clean build flake run

SHELL := /bin/bash
IMAGE_NAME := uncmath25/jupyter-notebook
IMAGE_HOME_DIR := /home/jovyan

clean:
	@echo "*** Cleaning the repo ***"
	rm -rf output/*

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
	docker run \
		--rm \
		-p 8888:8888 \
		-e JUPYTER_ENABLE_LAB=yes \
		-v "$$(pwd)/notebooks:$(IMAGE_HOME_DIR)/notebooks" \
		$(IMAGE_NAME)
