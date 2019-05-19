.PHONY: setup clean build flake down upn

DOCKER_COMPOSE_NOTEBOOK="docker-compose-notebook.yml"

setup:
	@echo "*** Setting up the repo ***"
	chmod +x ./bin/setup_repo.sh; \
	./bin/setup_repo.sh

clean:
	@echo "*** Cleaning the repo ***"
	rm -rf output/*

build:
	@echo "*** Building the docker images ***"
	docker-compose -f $(DOCKER_COMPOSE_NOTEBOOK) build
	docker build -f "Dockerfile-scripts" -t "pythonprototypes_scripts" .

flake:
	@echo "*** Linting the python scripts ***"
	./bin/flake_repo.sh

down:
	@echo "*** Stopping the Dockerized environments ***"
	docker-compose -f $(DOCKER_COMPOSE_NOTEBOOK) down --remove-orphans

upn: down
	@echo "*** Starting the Dockerized jupyter server ***"
	@echo "Access the server at: http://localhost:8888"
	docker-compose -f $(DOCKER_COMPOSE_NOTEBOOK) up -d
