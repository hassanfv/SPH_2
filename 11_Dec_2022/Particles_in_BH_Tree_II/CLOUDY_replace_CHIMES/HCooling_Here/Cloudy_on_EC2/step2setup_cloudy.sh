#!/bin/bash

# Define arrays for public DNS names of the instances
PUBLIC_DNS=(
    "ec2-18-195-51-116.eu-central-1.compute.amazonaws.com"
    "ec2-3-126-146-96.eu-central-1.compute.amazonaws.com"
    "ec2-3-72-72-61.eu-central-1.compute.amazonaws.com"
)

# SSH key for AWS instances
AWS_KEY="keyTest1.pem"

# Loop over the instances to perform the setup
for dns in "${PUBLIC_DNS[@]}"; do
    ssh -o StrictHostKeyChecking=no -i "$AWS_KEY" ubuntu@"$dns" <<'EOF'
    # Update package lists and install dependencies
    sudo apt-get update -y
    sudo apt-get install -y python3-mpi4py python3-pip git-lfs cmake libhdf5-serial-dev build-essential perl

    # Install Python packages
    pip3 install numpy pandas h5py

    # Download, extract, and compile Cloudy (To speed up download "cloudy-master.tar.gz" yourself !!!)
    #curl -O https://gitlab.nublado.org/cloudy/cloudy/-/archive/master/cloudy-master.tar.gz
    tar xvf cloudy-master.tar.gz
    cd cloudy-master/source/
    make
    cd ~/

    # Set up the binary and script for execution
    #mkdir -p .my_bin
    #echo '/home/ubuntu/cloudy-master/source/cloudy.exe -p \$1' > .my_bin/cloudy
    #chmod +x .my_bin/cloudy
    #echo "export PATH=\"/home/ubuntu/.my_bin:\$PATH\"" >> ~/.bashrc

    # Source .bashrc to update the current session (note: may not be effective in this non-interactive shell)
    source ~/.bashrc
EOF
done

