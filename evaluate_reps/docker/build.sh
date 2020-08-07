#!/bin/sh

#
DOCKER_CONTAINER_NAME=merchant-embedding #Example koho-transfers-api-dev
DOCKER_CONTAINER_IMAGE_NAME=merchant-embedding-img #Example koho-transfers-api-dev-img

CONTAINER_DIRECTORY=`pwd`/container

PROJECT_DIRECTORY=`pwd`/container/app/
DOCKER_PROJECT_DIRECTORY=/home/

TAG=$1

# Stop and remove the previous docker container (if it exists).
docker stop "${DOCKER_CONTAINER_NAME}"
docker rm   "${DOCKER_CONTAINER_NAME}"


# Build the docker container.
docker build -f container/Dockerfile -t "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" "${CONTAINER_DIRECTORY}"


# Run the (newly built) docker container.
# Open port 6006 so that TensorBoard can be used to visualize training.
docker run -it                                                        \
    -v "${PROJECT_DIRECTORY}:${DOCKER_PROJECT_DIRECTORY}"    \
    --name "${DOCKER_CONTAINER_NAME}" "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" \
