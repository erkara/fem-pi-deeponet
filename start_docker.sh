#!/bin/bash
# we need Docker
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

#check gpu if not fall back cpu(slow for sure)
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    echo "GPU detected — running with GPU support"
    GPU_FLAG="--gpus all"
else
    echo "No GPU detected — running on CPU only"
    GPU_FLAG=""
fi

# image hosting all firedrake+jax+Pytorch stuff
docker pull erdi28/firedrake-deeponet:latest

# run locally
sudo docker run $GPU_FLAG -it \
  -v $(pwd):/workspace \
  -w /workspace \
  -e LOCAL_UID=$(id -u) \
  -e LOCAL_GID=$(id -g) \
  --user $(id -u):$(id -g) \
  erdi28/firedrake-deeponet:latest \
  bash -c '
    echo ">>> Checking GPU availability..."
    echo "--- PyTorch ---"
    python3 -c "import torch; print(\"torch version:\", torch.__version__); print(\"CUDA available:\", torch.cuda.is_available()); print(\"torch.version.cuda:\", torch.version.cuda)"
    echo "--- JAX ---"
    python3 -c "import jax; print(\"jax version:\", jax.__version__); print(\"Devices:\", jax.devices())"
    exec bash
  '

