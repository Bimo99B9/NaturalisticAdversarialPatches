# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade torch torchvision

# Copy your entire project
COPY . /usr/src/app/

# Upgrade the packaging tools
RUN pip install --upgrade pip setuptools wheel

# First, install dependencies for your own scripts (like torchsummary)
RUN pip install -r requirements.txt

# Now, install the ultralytics package itself in editable mode
RUN pip install -e ./ultralytics

# Define environment variable (optional)
ENV NAME="Naturalistic-Adversarial-Patch"

# Define the default command
CMD ["python", "ensemble.py"]