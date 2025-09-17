#!/bin/bash

# Exit immediately if any command fails
set -e

# Define the target directory for clarity
TARGET_DIR="./GANLatentDiscovery/models"
OUTPUT_FILE="pretrained_biggan.tar"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Download the file, skipping if it already exists and is complete
echo "Downloading BigGAN pretrained weights..."
wget -c "https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar" -O "$OUTPUT_FILE"

# Extract the archive directly into the target directory.
# The -C flag tells tar to change to that directory before extracting.
# This will overwrite existing files safely.
echo "Extracting weights to $TARGET_DIR..."
tar xf "$OUTPUT_FILE" -C "$TARGET_DIR"

# Clean up the downloaded tar file
echo "Cleaning up..."
rm "$OUTPUT_FILE"

echo "✅ Done. Weights are installed."