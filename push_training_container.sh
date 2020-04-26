#!/bin/sh

TAG=$1
MODEL_DIR=$2


DOCKER_CONTAINER_NAME=user-embedding #Example koho-transfers-api-dev
DOCKER_CONTAINER_IMAGE_NAME=user-embedding-img #Example koho-transfers-api-dev-img

CONTAINER_DIRECTORY=`pwd`/$MODEL_DIR

PROJECT_DIRECTORY=`pwd`/$MODEL_DIR/app/
DOCKER_PROJECT_DIRECTORY=/home/

# Stop and remove the previous docker container (if it exists).
docker stop "${DOCKER_CONTAINER_NAME}"
docker rm   "${DOCKER_CONTAINER_NAME}"


# Build the docker container.
docker build -f $MODEL_DIR/Dockerfile.remote     \
             --build-arg EXP_ID=$TAG \
             -t "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" "${CONTAINER_DIRECTORY}"

docker tag "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" gcr.io/tensile-oarlock-191715/"${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}"
docker push gcr.io/tensile-oarlock-191715/"${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}"
