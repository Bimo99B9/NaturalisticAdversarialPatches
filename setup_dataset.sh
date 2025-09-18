#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Setup ---
# Create the dataset directory and navigate into it.
echo "Creating dataset directory..."
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
echo "Downloading original labels..."
wget https://www.dropbox.com/s/fjt76tcl9flurpb/test_inria_label.zip?dl=1 -O test_label.zip
wget https://www.dropbox.com/s/q0riodfy6bk9esk/train_inria_label.zip?dl=1 -O train_label.zip

echo "Extracting all labels..."
# Use -o to overwrite files without prompting.
unzip -oq test_label.zip
unzip -oq train_label.zip
# Unzip new labels from the parent directory into a dedicated 'new_labels' folder.
unzip -oq ../new_labels.zip -d new_labels

# --- Organize Files ---
echo "Copying all labels to dataset folders..."
# Copy original downloaded labels
cp -r train_inria_label/* inria/Train/pos/
cp -r test_inria_label/* inria/Test/pos/
# Copy new local labels (path is now correct)
cp -r new_labels/Train/* inria/Train/pos/
cp -r new_labels/Test/* inria/Test/pos/

# --- Clean Up ---
echo "Cleaning up archives and temporary directories..."
# Remove downloaded zip/tar files
rm inria.tar test_label.zip train_label.zip
# Remove all temporary extracted label directories
rm -r train_inria_label test_inria_label new_labels

echo "✅ Done. Dataset is ready."