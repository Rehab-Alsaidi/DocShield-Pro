#!/bin/bash
# Railway build script for CPU-optimized PyTorch installation

echo "🚂 Installing CPU-optimized dependencies for Railway..."

# Install PyTorch CPU version first
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
pip install -r requirements.txt --no-deps

echo "✅ Installation complete!"