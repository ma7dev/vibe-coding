#!/bin/bash

echo "🚀 Setting up Gemma-3 270M GGUF Saudi Benchmark Environment"
echo "================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
fi

echo "✓ Virtual environment ready"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version is sufficient for 270M model)
echo "🔄 Installing PyTorch..."
pip install torch torchvision torchaudio

# Install other requirements
echo "🔄 Installing other dependencies..."
pip install transformers>=4.36.0
pip install pandas>=1.5.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0
pip install peft>=0.7.0
pip install trl>=0.7.0
pip install bitsandbytes>=0.41.0

# Install Unsloth
echo "🔄 Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Clone the benchmark dataset if not already present
if [ ! -d "Pico-Saudi-LLMs-Benchmark" ]; then
    echo "🔄 Cloning Pico-Saudi-LLMs-Benchmark..."
    git clone https://github.com/mznmel/Pico-Saudi-LLMs-Benchmark.git
else
    echo "✓ Benchmark dataset already present"
fi

echo "================================================="
echo "✅ Setup completed successfully!"
echo "📋 To run the benchmark:"
echo "   source venv/bin/activate  # Activate virtual environment"
echo "   python gemma3_saudi_benchmark.py"
echo ""
echo "📁 Files created:"
echo "   - gemma3_saudi_benchmark.py (main script)"
echo "   - requirements.txt (dependencies)"
echo "   - setup.sh (this setup script)"
echo "   - venv/ (virtual environment)"
echo "   - Pico-Saudi-LLMs-Benchmark/ (dataset)"