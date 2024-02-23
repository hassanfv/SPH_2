#!/bin/bash

# Define arrays for public DNS names and private IPs of the instances
PUBLIC_DNS=(
    "ec2-18-195-51-116.eu-central-1.compute.amazonaws.com"
    "ec2-3-126-146-96.eu-central-1.compute.amazonaws.com"
    "ec2-3-72-72-61.eu-central-1.compute.amazonaws.com"
)

PRIVATE_IPS=(
    "172.31.39.62"
    "172.31.33.131"
    "172.31.45.28"
)

# SSH key for AWS instances
AWS_KEY="keyTest1.pem"

# Loop over the instances to perform initial setup
for i in {0..2}; do
    ssh -o StrictHostKeyChecking=no -i "$AWS_KEY" ubuntu@"${PUBLIC_DNS[$i]}" <<'EOF'
sudo apt-get update
sudo apt-get install -y python3-mpi4py
sudo apt install -y python3-pip
pip3 install numpy numba
echo -e "172.31.39.62\n172.31.33.131\n172.31.45.28" > hosts
EOF
done

# Key to be copied to all instances
KEY_TO_COPY="keyTest1.pem"

# Copy the key to each instance
for dns in "${PUBLIC_DNS[@]}"; do
    scp -i "$AWS_KEY" "$KEY_TO_COPY" ubuntu@"$dns":~/.ssh/id_rsa
done



# Add host keys to known_hosts file
for dns in "${PUBLIC_DNS[@]}"; do
    ssh-keyscan -H $dns >> ~/.ssh/known_hosts
done

# Loop to SSH from each instance to every other instance and back
for i in {0..2}; do
    for j in {0..2}; do
        if [ $i -ne $j ]; then
            # SSH into instance i, then SSH to instance j
            ssh -i "$AWS_KEY" ubuntu@"${PUBLIC_DNS[$i]}" <<EOF
                ssh -o StrictHostKeyChecking=no '${PRIVATE_IPS[$j]}'
                # Perform operations in instance j or wait for user to exit manually
                # After exiting from instance j, you will be back in instance i
EOF
            # If you want to automate exiting from instance i as well, you can include commands for that
        fi
    done
done




# Copy Archive.zip to all instances
ARCHIVE="Archive.zip"
for dns in "${PUBLIC_DNS[@]}"; do
    scp -i "$AWS_KEY" "$ARCHIVE" ubuntu@"$dns":~/
done

# Install zip, unzip Archive.zip on each instance
for dns in "${PUBLIC_DNS[@]}"; do
    ssh -i "$AWS_KEY" ubuntu@"$dns" <<'EOF'
sudo apt-get install -y zip unzip
unzip Archive.zip
EOF
done


# Choose one instance to run the mpirun command from, using the hosts file you created earlier
# Assuming you use the first instance in your PUBLIC_DNS array and the total number of CPUs is 8
ssh -i "$AWS_KEY" ubuntu@"${PUBLIC_DNS[0]}" <<'EOF'
mpirun --hostfile hosts -np 3 python3.8 mpi_test.py
EOF

