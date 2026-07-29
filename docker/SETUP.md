# Docker Setup Guide

This guide explains how to restore and run the pre-built Docker image for the Jetson dehazing project.

## Prerequisites

- Docker installed on the Jetson device
- NVIDIA Container Runtime installed (required for GPU access)
- The exported image archive: `my_jetson_env.tar`

---

## Step 1 — Load the Docker Image

```bash
docker load -i my_jetson_env.tar
```

Expected output:

```
Loaded image: my_jetson_env:trt_ready
```

Verify the image is available:

```bash
docker images
```

You should see:

```
REPOSITORY      TAG         IMAGE ID
my_jetson_env   trt_ready   xxxxxxxxxxxx
```

---

## Step 2 — Create the Workspace Directory

Create the host directory that will be mounted into the container:

```bash
mkdir -p ~/jetson_projects
```

Anything placed in this folder will be accessible inside the container at `/workspace`.

---

## Step 3 — Run the Container

Use the provided script from the repo root:

```bash
bash run_container.sh
```

Or run manually:

```bash
docker run -it \
    --rm \
    --name jetson_dev \
    --runtime=nvidia \
    --network host \
    --device /dev/video0 \
    -v ~/jetson_projects:/workspace \
    my_jetson_env:trt_ready \
    /bin/bash
```

### Flag Reference

| Flag | Purpose |
|---|---|
| `--rm` | Automatically remove the container on exit |
| `--name jetson_dev` | Assign a human-readable container name |
| `--runtime=nvidia` | Enable Jetson GPU access |
| `--network host` | Share the host network (needed for Flask streaming) |
| `--device /dev/video0` | Pass the camera device into the container |
| `-v ~/jetson_projects:/workspace` | Mount the project folder |

---

## Step 4 — Verify GPU Access

Inside the container:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

Or check CUDA directly:

```bash
ls /usr/local/cuda
```

---

## Step 5 — Verify Camera Access

Inside the container:

```bash
ls /dev/video*
# Expected: /dev/video0
```

---

## Step 6 — Reconnect to a Running Container

If the container is already running:

```bash
docker exec -it jetson_dev /bin/bash
```

---

## Step 7 — Stop the Container

From inside the container:

```bash
exit
```

Or from another terminal on the host:

```bash
docker stop jetson_dev
```

---

## Troubleshooting

### Image not found after `docker load`

```bash
docker images | grep my_jetson_env
```

If not present, re-run:

```bash
docker load -i my_jetson_env.tar
```

### GPU runtime error

Verify the NVIDIA runtime is registered:

```bash
docker info | grep Runtime
# Should include: nvidia
```

### Camera not detected (`/dev/video0` missing)

Check on the **host** first:

```bash
ls /dev/video*
```

If `/dev/video0` does not exist on the host, the container cannot access it. Check USB camera connections.

---

## Quick Reference

```bash
# Load image
docker load -i my_jetson_env.tar

# Run container
bash run_container.sh
```
