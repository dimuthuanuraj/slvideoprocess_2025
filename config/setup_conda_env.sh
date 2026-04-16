#!/bin/bash
# Conda Environment Setup Script for SLCeleb Video Processing (Modern Version)
# Date: October 31, 2025
# Python Version: 3.10

set -e  # Exit on error

echo "=========================================="
echo "Setting up SLCeleb Video Processing Environment"
echo "=========================================="
echo ""

# Environment name
ENV_NAME="slceleb_modern"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Source conda to enable conda commands
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment already exists
ENV_EXISTS=false
if conda env list | grep -q "^${ENV_NAME} "; then
    ENV_EXISTS=true
    echo "✓ Environment '${ENV_NAME}' already exists."
    echo "  Using existing environment and checking installed packages..."
    echo ""
else
    echo "Creating new conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo ""
fi

echo "Activating environment..."
conda activate ${ENV_NAME}

echo ""
echo "✓ Environment activated: $(which python)"
echo "✓ Python version: $(python --version)"
echo ""

# Function to check if a package is installed
check_package() {
    python -c "import $1" &> /dev/null
}

# Function to check if a conda package is installed
check_conda_package() {
    conda list | grep -q "^$1 "
}

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "=========================================="
echo "Installing Core Scientific Libraries"
echo "=========================================="
CORE_PACKAGES=(numpy scipy scikit-learn pandas)
MISSING_CORE=()
for pkg in "${CORE_PACKAGES[@]}"; do
    if ! check_conda_package "$pkg"; then
        MISSING_CORE+=("$pkg")
    else
        echo "✓ $pkg already installed"
    fi
done

if [ ${#MISSING_CORE[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_CORE[*]}"
    conda install -y "${MISSING_CORE[@]}" -c conda-forge
else
    echo "✓ All core scientific libraries already installed"
fi

echo ""
echo "=========================================="
echo "Installing Computer Vision Libraries"
echo "=========================================="

# Check OpenCV (use pip instead of conda for better Python version compatibility)
if ! check_package "cv2"; then
    echo "Installing opencv-python and opencv-contrib-python..."
    pip install opencv-python>=4.8.0
    pip install opencv-contrib-python>=4.8.0
else
    echo "✓ opencv already installed"
fi

# Check MediaPipe
if ! check_package "mediapipe"; then
    echo "Installing mediapipe..."
    pip install mediapipe>=0.10.8
else
    echo "✓ mediapipe already installed"
fi

# Check Pillow
if ! check_package "PIL"; then
    echo "Installing Pillow..."
    pip install Pillow>=10.0.0
else
    echo "✓ Pillow already installed"
fi

# Check imageio
if ! check_package "imageio"; then
    echo "Installing imageio..."
    pip install imageio>=2.31.0
else
    echo "✓ imageio already installed"
fi

echo ""
echo "=========================================="
echo "Installing Deep Learning Frameworks"
echo "=========================================="

# Check if PyTorch is already installed
PYTORCH_INSTALLED=false
if check_package "torch"; then
    PYTORCH_INSTALLED=true
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    echo "✓ PyTorch already installed (version: $TORCH_VERSION)"
    echo "  CUDA available: $CUDA_AVAILABLE"
fi

# Detect if CUDA is available
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    # Check if nvidia-smi works (no driver mismatch)
    if nvidia-smi &> /dev/null; then
        echo "✓ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader
        GPU_AVAILABLE=true
    else
        echo "⚠ NVIDIA driver detected but not working properly (possible driver/library mismatch)"
        echo "  This is usually caused by kernel driver not matching CUDA library version"
        echo "  To fix: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
        echo ""
        echo "  Proceeding with CPU-only installation for now..."
        GPU_AVAILABLE=false
    fi
else
    echo "⚠ No NVIDIA GPU detected"
    GPU_AVAILABLE=false
fi

echo ""
# Only install PyTorch if not already installed or if GPU availability has changed
if [ "$PYTORCH_INSTALLED" = false ]; then
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Installing PyTorch with CUDA support..."
        conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    else
        echo "Installing PyTorch CPU-only..."
        conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
    fi
elif [ "$GPU_AVAILABLE" = true ] && [ "$CUDA_AVAILABLE" = "False" ]; then
    echo "⚠ GPU detected but PyTorch doesn't have CUDA support"
    read -p "Do you want to reinstall PyTorch with CUDA support? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Reinstalling PyTorch with CUDA support..."
        conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
    fi
else
    echo "✓ PyTorch installation is appropriate for current hardware"
fi

# Check ONNX Runtime
if ! check_package "onnxruntime" && ! python -c "import onnxruntime" &> /dev/null; then
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Installing ONNX Runtime with GPU support..."
        pip install onnxruntime-gpu>=1.16.0
    else
        echo "Installing ONNX Runtime (CPU)..."
        pip install onnxruntime>=1.16.0
    fi
else
    echo "✓ ONNX Runtime already installed"
fi

echo ""
echo "=========================================="
echo "Installing Face Recognition Libraries"
echo "=========================================="

if ! check_package "insightface"; then
    echo "Installing insightface..."
    pip install insightface>=0.7.3
else
    echo "✓ insightface already installed"
fi

if ! check_package "onnx"; then
    echo "Installing onnx..."
    pip install onnx>=1.15.0
else
    echo "✓ onnx already installed"
fi

echo ""
echo "=========================================="
echo "Installing Audio Processing Libraries"
echo "=========================================="

AUDIO_PACKAGES=(librosa soundfile)
MISSING_AUDIO=()
for pkg in "${AUDIO_PACKAGES[@]}"; do
    if ! check_conda_package "$pkg"; then
        MISSING_AUDIO+=("$pkg")
    else
        echo "✓ $pkg already installed"
    fi
done

if [ ${#MISSING_AUDIO[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_AUDIO[*]}"
    conda install -y -c conda-forge "${MISSING_AUDIO[@]}"
else
    echo "✓ All audio processing libraries already installed"
fi

if ! check_package "python_speech_features"; then
    echo "Installing python-speech-features..."
    pip install python-speech-features>=0.6
else
    echo "✓ python-speech-features already installed"
fi

echo ""
echo "=========================================="
echo "Installing Utilities"
echo "=========================================="

UTIL_PACKAGES=(tqdm matplotlib seaborn pyyaml)
MISSING_UTILS=()
for pkg in "${UTIL_PACKAGES[@]}"; do
    if ! check_conda_package "$pkg"; then
        MISSING_UTILS+=("$pkg")
    else
        echo "✓ $pkg already installed"
    fi
done

if [ ${#MISSING_UTILS[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_UTILS[*]}"
    conda install -y -c conda-forge "${MISSING_UTILS[@]}"
else
    echo "✓ All utilities already installed"
fi

echo ""
echo "=========================================="
echo "Installing Development Tools (Optional)"
echo "=========================================="

DEV_TOOLS=(pytest black flake8 ipython)
for tool in "${DEV_TOOLS[@]}"; do
    if ! pip show "$tool" &> /dev/null; then
        echo "Installing $tool..."
        case $tool in
            pytest) pip install pytest>=7.4.0 ;;
            black) pip install black>=23.7.0 ;;
            flake8) pip install flake8>=6.1.0 ;;
            ipython) pip install ipython>=8.14.0 ;;
        esac
    else
        echo "✓ $tool already installed"
    fi
done

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "Installed packages summary:"
python -c "
import sys
packages = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'opencv': 'cv2',
    'mediapipe': 'mediapipe',
    'torch': 'torch',
    'insightface': 'insightface',
    'librosa': 'librosa'
}

print('')
for name, module in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {name}: {version}')
    except ImportError:
        print(f'  ✗ {name}: NOT INSTALLED')
"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Test MediaPipe face tracking:"
echo "   python mediapipe_face_tracking.py"
echo ""
echo "3. Run the main pipeline (after integration):"
echo "   python run.py --POI <speaker_name>"
echo ""
echo "4. To deactivate:"
echo "   conda deactivate"
echo ""
echo "5. To remove this environment:"
echo "   conda env remove -n ${ENV_NAME}"
echo ""
echo "=========================================="
echo "Environment setup completed successfully!"
echo "=========================================="
