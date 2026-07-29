#!/bin/bash
# run_container.sh
#
# Launches the Jetson TensorRT dehazing Docker container.
# Passes through the GPU and camera device.
#
# Usage:
#   bash run_container.sh

IMAGE="my_jetson_env:trt_ready"
CONTAINER_NAME="jetson_dev"

echo "Starting container: ${CONTAINER_NAME}"
echo "Image            : ${IMAGE}"
echo ""

docker run -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --network host \
    --device /dev/video0 \
    "${IMAGE}" /bin/bash
