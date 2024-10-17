#!/bin/bash

# Stop containers
sudo docker compose --profile ssl down

# Update package lists
sudo apt update

# Upgrade packages
sudo apt upgrade -y

# Autoremove unused packages
sudo apt autoremove -y

# Clean up package cache
sudo apt autoclean

# Renew Let's Encrypt SSL certificate
yes | certbot --cert-path ./web_server/certs --key-path ./web_server/certs --fullchain-path ./web_server/certs --chain-path ./web_server/certs renew

# Define the repository directory
REPO_DIR="$HOME/based_camp_elastic_search"

# Check if the directory exists
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    echo "Navigated to $REPO_DIR"

    # Fetch the latest changes from the remote repository
    git fetch

    # Pull the latest changes
    git pull origin main

    echo "Repository updated successfully."
else
    echo "Directory $REPO_DIR does not exist. Please clone the repository first."
fi

# Start containers
docker compose --env-file env/prod.env --profile ssl up -d
