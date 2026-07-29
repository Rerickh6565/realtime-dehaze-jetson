#!/bin/bash
# run_container.sh
#
# Launches the Jetson TensorRT dehazing Docker container.
# Mounts ~/jetson_projects as /workspace, passes through GPU and camera.
#
# Usage:
#   bash run_container.sh

IMAGE="my_jetson_env:trt_ready"
CONTAINER_NAME="jetson_dev"
WORKSPACE="${HOME}/jetson_projects"

echo "Starting container: ${CONTAINER_NAME}"
echo "Image            : ${IMAGE}"
echo "Workspace mount  : ${WORKSPACE} → /workspace"
echo ""

docker run -it \
    --rm \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --network host \
    --device /dev/video0 \
    -v "${WORKSPACE}:/workspace" \
    "${IMAGE}" /bin/bash
