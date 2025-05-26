# ReFaceNet

![Latest Reconstruction](outputs/latest_reconstruction.png)

ReFaceNet reconstructs a full 3D face from multiple occluded or cropped images of the same person. It intelligently combines visible face parts from different images to create a complete facial reconstruction, handling partial occlusions and cropped photos.

## How It Works
- **Smart Face Detection**: Detects faces even when partially occluded by hands or objects
- **Landmark Alignment**: Uses facial landmarks to align images geometrically  
- **Intelligent Blending**: Combines the best visible parts from each input image
- **Quality Weighting**: Prioritizes clearer, higher-quality facial regions

## Latest Features
- Multi-method landmark detection with fallbacks
- Robust alignment validation to prevent distortions
- Feature-specific face masking (eyes, nose, mouth)
- Comprehensive logging and debugging output
- Automatic generation numbering for tracking improvements

## Quick Start

### 1. Setup Environment
```bash
# Follow setup instructions in SETUP_GUIDE.md
conda create -y -n ReFaceNet python=3.10
conda activate ReFaceNet
pip install opencv-python mediapipe tqdm
```

### 2. Add Your Images
Put your face images (can be partially occluded) in `data/raw/`

### 3. Run Reconstruction
```bash
python run_pipeline_smart.py
```

### 4. View Results
- `outputs/latest_reconstruction.png` - Your reconstructed face
- `outputs/debug/` - Debug visualizations
- `face_reconstruction.log` - Detailed processing log

## Output Files
- `generation_X_reconstruction.png` - Numbered results for tracking progress
- `latest_reconstruction.png` - Always the most recent result
- `generation_X_stats.json` - Processing statistics for each run
- `debug/` folder - Individual aligned images and landmark visualizations
