#!/bin/sh

#
DOCKER_CONTAINER_NAME=kfp_container #Example koho-transfers-api-dev
DOCKER_CONTAINER_IMAGE_NAME=kfp_container_img #Example koho-transfers-api-dev-img

CONTAINER_DIRECTORY=`pwd`/kfp_container

PROJECT_DIRECTORY=`pwd`/kfp_container/app/
DOCKER_PROJECT_DIRECTORY=/home/

EXP_ID=$1


# Stop and remove the previous docker container (if it exists).
docker stop "${DOCKER_CONTAINER_NAME}"
docker rm   "${DOCKER_CONTAINER_NAME}"

# Build the docker container.
docker build -f kfp_container/Dockerfile \
             --build-arg EXP_ID=$EXP_ID \
             -t "${DOCKER_CONTAINER_IMAGE_NAME}":"latest" "${CONTAINER_DIRECTORY}"

# Run the (newly built) docker container.
# Open port 6006 so that TensorBoard can be used to visualize training.
docker run -it                                            \
    -v "${PROJECT_DIRECTORY}:${DOCKER_PROJECT_DIRECTORY}"    \
    --name "${DOCKER_CONTAINER_NAME}" "${DOCKER_CONTAINER_IMAGE_NAME}":"latest" \
