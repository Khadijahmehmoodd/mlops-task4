# Variables
PYTHON_INTERPRETER = python3
DOCKER_IMAGE_NAME =  mlopstasks
DOCKER_CONTAINER_NAME = mlops-classtask-4

.PHONY: build run clean test

build:
    @echo "Building the Docker image..."
    docker build -t $(DOCKER_IMAGE_NAME) .

run:
    @echo "Running the Docker container..."
    docker run -it --rm --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)

clean:
    @echo "Cleaning up..."
    docker stop $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
    docker rm $(DOCKER_CONTAINER_NAME) 2>/dev/null || true

test:
    @echo "Running tests..."
    $(PYTHON_INTERPRETER) test.py
