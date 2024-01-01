#!/bin/bash

# Update and install necessary packages
sudo apt-get update -y
sudo apt-get install -y python3-mpi4py python3-pip git-lfs cmake libhdf5-serial-dev
pip3 install numpy h5py

# Clone repositories
git clone https://bitbucket.org/richings/chimes
git clone https://bitbucket.org/richings/chimes-data
git clone https://bitbucket.org/richings/chimes-driver

# Git LFS setup
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
git lfs install

# Sundials installation
wget https://github.com/LLNL/sundials/releases/download/v5.1.0/sundials-5.1.0.tar.gz
tar -zxvf sundials-5.1.0.tar.gz
cd sundials-5.1.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir/ ..
make
sudo make install

# Modify .bashrc
echo "export PATH=/path/to/install/dir/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/path/to/install/dir/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Copy Makefile to chimes directory. REMEMBER to place Makefile in the same folder as this file, i.e. setup.sh
cp ./Makefile ~/chimes

# Build chimes
cd ~/chimes
make

