#!/bin/bash

# EC2 Instance Public DNS Addresses
PUBLIC_DNS=(
    "ec2-3-126-152-108.eu-central-1.compute.amazonaws.com"
    "ec2-52-59-244-141.eu-central-1.compute.amazonaws.com"
    "ec2-18-193-71-17.eu-central-1.compute.amazonaws.com"
)

# SSH Key for AWS Instances
AWS_KEY="keyXtest.pem"

# Loop through each Public DNS Address
for dns in "${PUBLIC_DNS[@]}"; do
    echo "Processing $dns..."

    # Execute the original setup.sh operations on each EC2 instance
    ssh -i "$AWS_KEY" "ubuntu@$dns" "
        sudo apt-get update -y
        sudo apt-get install -y python3-mpi4py python3-pip git-lfs cmake libhdf5-serial-dev
        pip3 install numpy h5py
        git clone https://bitbucket.org/richings/chimes
        git clone https://bitbucket.org/richings/chimes-data
        git clone https://bitbucket.org/richings/chimes-driver
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        git lfs install
        wget https://github.com/LLNL/sundials/releases/download/v5.1.0/sundials-5.1.0.tar.gz
        tar -zxvf sundials-5.1.0.tar.gz
        cd sundials-5.1.0
        mkdir build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir/ ..
        make
        sudo make install
        echo 'export PATH=/path/to/install/dir/bin:\$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/path/to/install/dir/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc
        cp /home/ubuntu/Makefile /home/ubuntu/chimes
        cd /home/ubuntu/chimes
        make
    "
done

echo "Setup completed for all instances."

