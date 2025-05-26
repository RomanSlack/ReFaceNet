# ReFaceNet Setup Guide

## Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended)
- CMake (for dlib compilation)

## Step 1: Install System Dependencies
```bash
# Install cmake for dlib
sudo apt update && sudo apt install -y cmake

# Or on macOS:
# brew install cmake
```

## Step 2: Environment Setup
```bash
# Create and activate conda environment
conda create -y -n ReFaceNet python=3.10
conda activate ReFaceNet

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python mediapipe trimesh open3d imageio tqdm Pillow numpy

# Install face detection library (requires cmake)
pip install face_recognition

# Install LaMa inpainting (optional, for original pipeline)
pip install simple-lama-inpainting
```

## Step 3: Setup DECA (3D Face Reconstruction)
```bash
cd DECA
pip install -r requirements.txt
python setup.py install

# Download DECA models (~350MB)
bash fetch_data.sh
# You'll need to register at https://flame.is.tue.mpg.de/
```

## Step 4: Prepare Your Images
1. Put your face images in `data/raw/` directory
2. Images should contain faces (can be partially occluded)
3. All images should be of the same person

## Step 5: Run Face Reconstruction

### Option A: Smart Pipeline (Recommended for occluded faces)
```bash
python run_pipeline_smart.py
```

### Option B: Basic Pipeline  
```bash
python run_pipeline.py
```

## Expected Outputs
- `outputs/smart_reconstruction.png` - AI-reconstructed face from partial views
- `outputs/debug/` - Debug images showing detection, landmarks, masks
- `outputs/processing_stats.json` - Detailed analysis of each input image
- `face_reconstruction.log` - Complete processing log

## Troubleshooting

### Face Detection Issues
- Ensure faces are clearly visible in images
- Try different lighting conditions
- Remove heavily occluded images

### CMake/dlib Installation Issues
```bash
# Alternative installation methods:
conda install -c conda-forge dlib
# or
pip install dlib-binary
```

### CUDA Issues
```bash
# CPU-only PyTorch if GPU unavailable:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Testing Your Setup
1. Run with test images: `python run_pipeline_improved.py`
2. Check output quality in `outputs/` folder
3. Report results back for further optimization