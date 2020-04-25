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
docker build -f $MODEL_DIR/Dockerfile.local     \
             --build-arg EXP_ID=$TAG \
             -t "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" "${CONTAINER_DIRECTORY}"

# Run the (newly built) docker container.
# Open port 5000
docker run -it -p "5000:5000"                                       \
    --env RUN_LOCALLY='True'   \
    -v "${PROJECT_DIRECTORY}:${DOCKER_PROJECT_DIRECTORY}"    \
    --name "${DOCKER_CONTAINER_NAME}" "${DOCKER_CONTAINER_IMAGE_NAME}":"${TAG}" \
