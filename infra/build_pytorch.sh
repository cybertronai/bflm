# usage: source build_nccl.sh
# Builds PyTorch, installs into current conda env.
# Use $NCCL_ROOT_DIR to customize NCCL
# Use existing conda env to minimize missing deps: conda create --name=pytorch_april_patched --clone pytorch_p36

cd ~
git clone --recursive https://github.com/pytorch/pytorch
cd ~/pytorch
git fetch
git checkout v1.1.0

# customize NCCL, stock version uses /usr/local/cuda/lib/libnccl.so.2

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
