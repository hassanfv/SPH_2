#!/bin/bash

# Update package lists
sudo apt-get update -y
sudo apt-get install -y python3-mpi4py python3-pip git-lfs cmake libhdf5-serial-dev
# Install GCC, G++, and GNU Make if not already installed
sudo apt-get install -y build-essential
pip3 install numpy pandas h5py

# Install Perl
apt-get install -y perl

# Step 1: Download the source code
curl -O https://gitlab.nublado.org/cloudy/cloudy/-/archive/master/cloudy-master.tar.gz

# Step 2: Extract the tarball
tar xvf cloudy-master.tar.gz

# Step 3: Change directory to the source folder
cd cloudy-master/source/

# Step 4: Compile the source code
make

# Step 5: Go back to the home directory
cd ~/

# Step 6: Create a new directory for the binary
mkdir -p .my_bin

# Step 7: Change directory to the newly created directory
cd .my_bin

# Step 8 & 9: Create the 'cloudy' script and write the command to execute cloudy.exe
echo '/home/ubuntu/cloudy-master/source/cloudy.exe -p $1' > cloudy

# Step 10: Make the script executable
chmod +x cloudy

# Step 11: Add the new directory to PATH in .bashrc
echo 'export PATH=/home/ubuntu/.my_bin:$PATH' >> ~/.bashrc

# Step 12: Source .bashrc to update the current session
source ~/.bashrc

# Step 13: Go back to the home directory
cd ~/

