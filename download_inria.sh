#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create the dataset directory only if it doesn't already exist.
mkdir -p dataset
cd dataset

# --- Download and Extract Main Dataset ---
# Use -c to continue downloading if interrupted, or skip if already complete.
echo "Downloading INRIA Person dataset..."
wget -c ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -O inria.tar
echo "Extracting dataset..."
tar xf inria.tar
mv INRIAPerson inria

# --- Download and Extract Labels ---
echo "Downloading labels..."
wget https://www.dropbox.com/s/fjt76tcl9flurpb/test_inria_label.zip?dl=1 -O test_label.zip
wget https://www.dropbox.com/s/q0riodfy6bk9esk/train_inria_label.zip?dl=1 -O train_label.zip

echo "Extracting labels..."
unzip -q test_label.zip
unzip -q train_label.zip

# --- Organize Files ---
echo "Copying labels to dataset folders..."
cp -r train_inria_label/* inria/Train/pos/
cp -r test_inria_label/* inria/Test/pos/

# --- Clean Up ---
echo "Cleaning up archive files..."
rm inria.tar test_label.zip train_label.zip

echo "✅ Done. Dataset is ready."