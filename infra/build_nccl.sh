# usage: source build_nccl.sh
# - builds NCCL from source into ~/nccl/build
# - changes default cuda to cuda-10
# - copies nccl libs to /usr/lib and /usr/local/cuda/lib, doesn't change nccl.2 symlinks

# NCCL locations
# /usr/lib/libnccl*   # default examples?
# /usr/local/cuda/lib/libncc*   # stock pytorch
# ~/nccl/build    # build from source
# $NCCL_HOME      # used by make


# NCCL links to /usr/local/cuda, make sure it's cuda-10.0
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda

git clone https://github.com/NVIDIA/nccl.git
cd ~/nccl
make -j src.build
sudo apt install -y build-essential devscripts debhelper
make pkg.debian.build

# move to /usr/lib, this is location where NCCL examples look for nccl
sudo cp /home/ubuntu/nccl/build/lib/libnccl* /usr/lib

# copy nccl 2.4.6, but the symlinks nccl and nccl.2 will point to nccl 2.3.7
sudo cp /home/ubuntu/nccl/build/lib/libnccl* /usr/local/cuda/lib

# (unneeded? Not clear if that path is used)
export NCCL_HOME=~/nccl/build/lib

# testing
cd ~
git clone https://github.com/NVIDIA/nccl-tests.git
cd ~/nccl-tests
make

export NCCL_DEBUG=VERSION
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 8

cd ~
