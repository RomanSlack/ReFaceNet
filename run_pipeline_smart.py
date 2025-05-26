import cv2, numpy as np, mediapipe as mp, glob, torch
import logging
from tqdm import tqdm
from pathlib import Path
import json
import time

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_reconstruction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs")
DEBUG_DIR = OUT_DIR / "debug"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

RES = 512

# Initialize MediaPipe with very low confidence for difficult cases
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.1)
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, 
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
    max_num_faces=1
)

def analyze_image_quality(img, face_bbox=None):
    """Advanced image quality analysis for face reconstruction"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Focus on face region if available
    if face_bbox:
        x, y, w, h = face_bbox
        face_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]
    else:
        face_gray = gray
        face_color = img
    
    # Calculate multiple quality metrics
    laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()  # Sharpness
    brightness = np.mean(face_gray)
    contrast = np.std(face_gray)
    
    # Additional quality metrics
    # Texture richness (gradient magnitude)
    grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    texture_score = np.mean(gradient_mag)
    
    # Color saturation in face region
    hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1])
    
    # Exposure quality (avoid over/under-exposed)
    exposure_score = 1.0 - (np.sum(face_gray > 240) + np.sum(face_gray < 15)) / face_gray.size
    
    # Combined quality score with weights
    quality_score = (
        (laplacian_var / 200) * 0.3 +      # Sharpness (30%)
        (texture_score / 50) * 0.25 +      # Texture detail (25%)
        (contrast / 80) * 0.2 +            # Contrast (20%)
        exposure_score * 0.15 +            # Exposure (15%)
        (saturation / 100) * 0.1           # Color richness (10%)
    )
    
    # Penalty for extreme brightness
    brightness_penalty = abs(brightness - 127) / 127
    quality_score *= (1 - brightness_penalty * 0.3)
    
    logger.info(f"   Sharpness: {laplacian_var:.1f}, Texture: {texture_score:.1f}, Contrast: {contrast:.1f}")
    logger.info(f"   Brightness: {brightness:.1f}, Saturation: {saturation:.1f}, Exposure: {exposure_score:.2f}")
    logger.info(f"   Quality Score: {quality_score:.3f}")
    
    return quality_score

def detect_face_with_logging(img, img_path):
    """Detect face with detailed logging"""
    logger.info(f"\n=== Processing: {Path(img_path).name} ===")
    logger.info(f"Image shape: {img.shape}")
    
    h, w = img.shape[:2]
    
    # Analyze image quality (will be updated with face_bbox later)
    quality_score = analyze_image_quality(img)
    
    # Try face detection
    start_time = time.time()
    results = mp_face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_time = time.time() - start_time
    
    logger.info(f"Face detection took: {detection_time:.3f}s")
    
    if not results.detections:
        logger.warning("❌ No faces detected")
        return None, quality_score
    
    logger.info(f"✅ Found {len(results.detections)} face(s)")
    
    # Find largest face
    faces_info = []
    for i, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box
        confidence = detection.score[0]
        area = bbox.width * bbox.height
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        faces_info.append({
            'id': i,
            'confidence': confidence,
            'area': area,
            'bbox': (x, y, width, height)
        })
        
        logger.info(f"   Face {i}: confidence={confidence:.3f}, area={area:.4f}, bbox=({x},{y},{width},{height})")
    
    # Select best face (highest confidence * area)
    best_face = max(faces_info, key=lambda x: x['confidence'] * x['area'])
    logger.info(f"Selected face {best_face['id']} (score: {best_face['confidence'] * best_face['area']:.4f})")
    
    return best_face['bbox'], quality_score

def extract_landmarks_with_logging(img, face_bbox, img_path):
    """Extract landmarks with detailed logging and fallback methods"""
    h, w = img.shape[:2]
    
    # Try multiple extraction methods
    methods_tried = []
    
    # Method 1: Use detected face bbox
    if face_bbox:
        x, y, width, height = face_bbox
        # Expand bbox
        pad = 30
        x = max(0, x - pad)
        y = max(0, y - pad)
        width = min(w - x, width + 2*pad)
        height = min(h - y, height + 2*pad)
        
        logger.info(f"Face region: ({x},{y}) -> ({x+width},{y+height}) (padded)")
        face_img = img[y:y+height, x:x+width]
        offset = (x, y)
        
        # Save debug image of face region
        debug_path = DEBUG_DIR / f"{Path(img_path).stem}_face_region.jpg"
        cv2.imwrite(str(debug_path), face_img)
        logger.info(f"Saved face region: {debug_path}")
        
        # Try landmark extraction on face region
        start_time = time.time()
        fh, fw = face_img.shape[:2]
        res = mp_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        landmark_time = time.time() - start_time
        
        logger.info(f"Method 1 (face region) took: {landmark_time:.3f}s")
        methods_tried.append("face_region")
        
        if res.multi_face_landmarks:
            pts = np.array([[lm.x*fw + offset[0], lm.y*fh + offset[1]] 
                           for lm in res.multi_face_landmarks[0].landmark])
            logger.info(f"✅ Method 1: Extracted {len(pts)} landmarks from face region")
            
            # Save debug image with landmarks
            debug_img = img.copy()
            for pt in pts:
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            
            debug_path = DEBUG_DIR / f"{Path(img_path).stem}_landmarks.jpg"
            cv2.imwrite(str(debug_path), debug_img)
            logger.info(f"Saved landmarks visualization: {debug_path}")
            
            return pts
    
    # Method 2: Try full image with very low confidence
    logger.info("Method 1 failed, trying full image...")
    start_time = time.time()
    
    # Create a more permissive face mesh detector
    mp_mesh_fallback = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, 
        min_detection_confidence=0.05,
        min_tracking_confidence=0.05,
        max_num_faces=1
    )
    
    res = mp_mesh_fallback.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmark_time = time.time() - start_time
    
    logger.info(f"Method 2 (full image, low confidence) took: {landmark_time:.3f}s")
    methods_tried.append("full_image_low_confidence")
    
    if res.multi_face_landmarks:
        pts = np.array([[lm.x*w, lm.y*h] for lm in res.multi_face_landmarks[0].landmark])
        logger.info(f"✅ Method 2: Extracted {len(pts)} landmarks from full image")
        
        # Save debug image with landmarks
        debug_img = img.copy()
        for pt in pts:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        
        debug_path = DEBUG_DIR / f"{Path(img_path).stem}_landmarks.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"Saved landmarks visualization: {debug_path}")
        
        return pts
    
    # Method 3: Try with image preprocessing
    logger.info("Method 2 failed, trying with image enhancement...")
    
    # Enhance contrast and brightness
    enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    
    start_time = time.time()
    res = mp_mesh_fallback.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    landmark_time = time.time() - start_time
    
    logger.info(f"Method 3 (enhanced image) took: {landmark_time:.3f}s")
    methods_tried.append("enhanced_image")
    
    if res.multi_face_landmarks:
        pts = np.array([[lm.x*w, lm.y*h] for lm in res.multi_face_landmarks[0].landmark])
        logger.info(f"✅ Method 3: Extracted {len(pts)} landmarks from enhanced image")
        
        # Save debug image with landmarks
        debug_img = img.copy()
        for pt in pts:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        
        debug_path = DEBUG_DIR / f"{Path(img_path).stem}_landmarks.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"Saved landmarks visualization: {debug_path}")
        
        return pts
    
    logger.warning(f"❌ All methods failed. Tried: {methods_tried}")
    return None

def create_face_mask(img, landmarks):
    """Create mask of visible face regions focusing on core facial features"""
    h, w = img.shape[:2]
    
    # Create a more conservative face mask using facial feature regions
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get face outline landmarks (more conservative than full convex hull)
    face_outline = landmarks[[10, 151, 9, 8, 168, 6, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
    
    # Create face region
    cv2.fillPoly(mask, [face_outline.astype(np.int32)], 255)
    
    # Create separate masks for key facial features
    # Left eye region
    left_eye_pts = landmarks[[33, 7, 163, 144, 145, 153, 154, 155, 133]]
    cv2.fillPoly(mask, [left_eye_pts.astype(np.int32)], 255)
    
    # Right eye region  
    right_eye_pts = landmarks[[362, 398, 384, 385, 386, 387, 388, 466, 263]]
    cv2.fillPoly(mask, [right_eye_pts.astype(np.int32)], 255)
    
    # Nose region
    nose_pts = landmarks[[1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234]]
    cv2.fillPoly(mask, [nose_pts.astype(np.int32)], 255)
    
    # Mouth region
    mouth_pts = landmarks[[61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]]
    cv2.fillPoly(mask, [mouth_pts.astype(np.int32)], 255)
    
    # Smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    return mask

def normalize_skin_tone(img, mask):
    """Normalize skin tone for consistent color across images"""
    # Extract skin region using mask
    skin_pixels = img[mask > 0]
    
    if len(skin_pixels) == 0:
        return img
    
    # Calculate current skin tone statistics
    skin_mean = np.mean(skin_pixels, axis=0)
    skin_std = np.std(skin_pixels, axis=0)
    
    # Target skin tone (neutral warm tone)
    target_mean = np.array([155, 120, 100], dtype=np.float32)  # BGR
    target_std = np.array([25, 20, 18], dtype=np.float32)
    
    # Apply color transfer to skin regions only
    normalized = img.copy().astype(np.float32)
    
    # Avoid division by zero
    skin_std = np.maximum(skin_std, 1.0)
    
    for c in range(3):
        # Normalize and shift color channel
        normalized[:,:,c] = (normalized[:,:,c] - skin_mean[c]) * (target_std[c] / skin_std[c]) + target_mean[c]
    
    # Blend normalized version with original using mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    result = normalized * mask_3ch + img.astype(np.float32) * (1 - mask_3ch)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def create_feature_weight_map(mask, landmarks, transform_matrix):
    """Create weight map that prioritizes important facial features"""
    h, w = mask.shape
    weight_map = np.ones((h, w), dtype=np.float32)
    
    # Transform landmarks to aligned space
    landmarks_hom = np.column_stack([landmarks, np.ones(len(landmarks))])
    transformed_landmarks = (transform_matrix @ landmarks_hom.T).T
    
    # Define feature regions with different weights
    feature_regions = {
        'eyes': {'landmarks': [33, 7, 163, 144, 145, 153, 154, 155, 133, 362, 398, 384, 385, 386, 387, 388, 466, 263], 'weight': 1.5},
        'nose': {'landmarks': [1, 2, 5, 4, 6, 19, 94, 125], 'weight': 1.3},  
        'mouth': {'landmarks': [61, 84, 17, 314, 405, 320, 307, 375], 'weight': 1.4},
        'eyebrows': {'landmarks': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], 'weight': 1.2}
    }
    
    for feature_name, feature_info in feature_regions.items():
        try:
            # Get transformed landmark points for this feature
            feature_landmarks = transformed_landmarks[feature_info['landmarks']]
            
            # Create convex hull around feature
            if len(feature_landmarks) > 2:
                hull = cv2.convexHull(feature_landmarks.astype(np.int32))
                feature_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(feature_mask, [hull], 1)
                
                # Apply feature weight
                weight_map += feature_mask * (feature_info['weight'] - 1.0)
        except:
            continue  # Skip if landmarks are invalid
    
    return weight_map

def apply_advanced_post_processing(composite, weight_map):
    """Advanced post-processing pipeline for final image refinement"""
    logger.info("Stage 1: Edge smoothing and blending...")
    
    # Create confidence mask from weight accumulation
    confidence_mask = np.clip(weight_map / np.max(weight_map), 0, 1)
    
    # Edge detection for transition smoothing
    edge_mask = cv2.Canny((confidence_mask * 255).astype(np.uint8), 30, 100)
    edge_mask = cv2.dilate(edge_mask, np.ones((3,3), np.uint8), iterations=2)
    edge_mask = cv2.GaussianBlur(edge_mask, (5,5), 0) / 255.0
    
    # Bilateral filter for noise reduction while preserving edges
    denoised = cv2.bilateralFilter(composite, 9, 75, 75)
    
    # Blend denoised version at low-confidence edges
    edge_mask_3ch = cv2.cvtColor((edge_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    refined = composite.astype(np.float32) * (1 - edge_mask_3ch) + denoised.astype(np.float32) * edge_mask_3ch
    
    logger.info("Stage 2: Color and contrast enhancement...")
    
    # Adaptive histogram equalization for better contrast
    lab = cv2.cvtColor(refined.astype(np.uint8), cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Subtle color balance adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=3)
    
    logger.info("Stage 3: Detail enhancement...")
    
    # Multi-scale unsharp masking for better detail
    # Fine details
    gaussian_fine = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    enhanced = cv2.addWeighted(enhanced, 1.3, gaussian_fine, -0.3, 0)
    
    # Medium details  
    gaussian_medium = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.15, gaussian_medium, -0.15, 0)
    
    logger.info("Stage 4: Final skin smoothing...")
    
    # Gentle skin smoothing in low-confidence areas
    low_conf_mask = (confidence_mask < 0.7).astype(np.float32)
    smoothed = cv2.bilateralFilter(enhanced, 15, 80, 80)
    
    # Apply skin smoothing only where confidence is low
    low_conf_mask_3ch = cv2.cvtColor((low_conf_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) / 255.0
    final = enhanced.astype(np.float32) * (1 - low_conf_mask_3ch * 0.4) + smoothed.astype(np.float32) * (low_conf_mask_3ch * 0.4)
    
    return np.clip(final, 0, 255).astype(np.uint8)

def clean_debug_folder():
    """Clean debug folder before starting"""
    if DEBUG_DIR.exists():
        import shutil
        shutil.rmtree(DEBUG_DIR)
        logger.info(f"🧹 Cleaned debug folder: {DEBUG_DIR}")
    DEBUG_DIR.mkdir(exist_ok=True, parents=True)

def get_next_generation_number():
    """Find next generation number for output files"""
    existing_files = list(OUT_DIR.glob("generation_*_reconstruction.png"))
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            continue
    
    return max(numbers) + 1 if numbers else 1

def main():
    logger.info("🚀 Starting Smart Face Reconstruction Pipeline")
    logger.info(f"Input directory: {RAW_DIR}")
    logger.info(f"Output directory: {OUT_DIR}")
    
    # Clean debug folder
    clean_debug_folder()
    
    # Get generation number
    generation = get_next_generation_number()
    logger.info(f"📝 Generation: {generation}")
    
    # Find all images
    image_files = sorted(glob.glob(str(RAW_DIR / "*")))
    logger.info(f"Found {len(image_files)} files")
    
    if len(image_files) == 0:
        logger.error("❌ No images found!")
        return
    
    # Process each image
    valid_images = []
    processing_stats = []
    
    logger.info("\n" + "="*50)
    logger.info("PHASE 1: ANALYZING IMAGES")
    logger.info("="*50)
    
    for i, img_path in enumerate(image_files):
        logger.info(f"\n--- Image {i+1}/{len(image_files)} ---")
        
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"❌ Could not load: {img_path}")
            continue
        
        # Detect face
        face_bbox, quality_score = detect_face_with_logging(img, img_path)
        
        if face_bbox is None:
            logger.warning(f"Skipping {Path(img_path).name} - no face detected")
            continue
        
        # Extract landmarks
        landmarks = extract_landmarks_with_logging(img, face_bbox, img_path)
        
        if landmarks is None:
            logger.warning(f"Skipping {Path(img_path).name} - no landmarks")
            continue
        
        # Re-analyze quality focusing on detected face region
        if face_bbox:
            quality_score = analyze_image_quality(img, face_bbox)
        
        # Create face mask
        face_mask = create_face_mask(img, landmarks)
        visible_area = np.sum(face_mask > 0)
        
        logger.info(f"Visible face area: {visible_area} pixels")
        logger.info(f"Final quality score: {quality_score:.3f}")
        
        # Save mask debug
        debug_path = DEBUG_DIR / f"{Path(img_path).stem}_mask.jpg"
        cv2.imwrite(str(debug_path), face_mask)
        
        stats = {
            'path': img_path,
            'quality_score': float(quality_score),
            'visible_area': int(visible_area),
            'face_bbox': [int(x) for x in face_bbox],
            'landmarks': [[float(x), float(y)] for x, y in landmarks]
        }
        
        valid_images.append((img_path, img, landmarks, face_mask, quality_score))
        processing_stats.append(stats)
        
        logger.info(f"✅ Successfully processed: {Path(img_path).name}")
    
    # Save processing stats with generation number
    stats_file = OUT_DIR / f"generation_{generation}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(processing_stats, f, indent=2)
    
    logger.info(f"\n✅ Phase 1 complete: {len(valid_images)}/{len(image_files)} images processed successfully")
    
    if len(valid_images) < 2:
        logger.error(f"❌ Need at least 2 valid images, got {len(valid_images)}")
        return
    
    # Phase 2: Reconstruction
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: FACE RECONSTRUCTION")
    logger.info("="*50)
    
    # Select template (best quality image)
    template_idx = max(range(len(valid_images)), key=lambda i: valid_images[i][4])
    template_path, template_img, template_landmarks, _, _ = valid_images[template_idx]
    
    logger.info(f"Using template: {Path(template_path).name}")
    
    # Completely new approach: standardized face template
    # Define a standard face template with normalized positions
    template_size = RES
    
    # Standard face proportions (normalized to template_size)
    eye_y = template_size * 0.35  # Eyes at 35% from top
    eye_spacing = template_size * 0.25  # Eye spacing
    center_x = template_size * 0.5
    
    # Standard template points
    standard_template = np.array([
        [center_x - eye_spacing/2, eye_y],  # Left eye center
        [center_x + eye_spacing/2, eye_y],  # Right eye center  
        [center_x, template_size * 0.55],   # Nose tip
        [center_x - template_size*0.15, template_size * 0.75],  # Left mouth corner
        [center_x + template_size*0.15, template_size * 0.75],  # Right mouth corner
    ], dtype=np.float32)
    
    logger.info(f"Using standardized face template")
    logger.info(f"Template eye distance: {np.linalg.norm(standard_template[0] - standard_template[1]):.1f}px")
    
    # Initialize accumulation
    accum = np.zeros((RES, RES, 3), np.float32)
    weight_accum = np.zeros((RES, RES), np.float32)
    
    successful_alignments = 0
    
    for i, (img_path, img, landmarks, face_mask, quality_score) in enumerate(valid_images):
        logger.info(f"\nAligning image {i+1}: {Path(img_path).name}")
        
        try:
            # Extract key facial points from landmarks
            # Left eye center (average of eye landmarks)
            left_eye_pts = landmarks[[33, 7, 163, 144, 145, 153, 154, 155, 133]]
            left_eye_center = np.mean(left_eye_pts, axis=0)
            
            # Right eye center  
            right_eye_pts = landmarks[[362, 398, 384, 385, 386, 387, 388, 466, 263]]
            right_eye_center = np.mean(right_eye_pts, axis=0)
            
            # Nose tip
            nose_tip = landmarks[1]
            
            # Mouth corners
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            
            # Create source points array
            src_points = np.array([
                left_eye_center,
                right_eye_center, 
                nose_tip,
                left_mouth,
                right_mouth
            ], dtype=np.float32)
            
            logger.info(f"Extracted facial key points")
            logger.info(f"Eye distance: {np.linalg.norm(left_eye_center - right_eye_center):.1f}px")
            
            # Align to standard template using similarity transform
            M, _ = cv2.estimateAffinePartial2D(src_points, standard_template)
            
            if M is None:
                logger.warning(f"❌ Failed to compute alignment for {Path(img_path).name}")
                warped_img = cv2.resize(img, (RES, RES))
                warped_mask = cv2.resize(face_mask, (RES, RES))
                logger.info(f"🔄 Using resize fallback - no alignment matrix")
            else:
                # Check transformation sanity
                det = np.linalg.det(M[:2, :2])
                scale = np.sqrt(det)
                
                logger.info(f"Transform scale: {scale:.2f}, determinant: {det:.2f}")
                
                if scale < 0.2 or scale > 5.0:
                    logger.warning(f"❌ Extreme scaling detected for {Path(img_path).name}")
                    warped_img = cv2.resize(img, (RES, RES))
                    warped_mask = cv2.resize(face_mask, (RES, RES))
                    logger.info(f"🔄 Using resize fallback - extreme scale")
                else:
                    # Apply transformation
                    warped_img = cv2.warpAffine(img, M, (RES, RES))
                    warped_mask = cv2.warpAffine(face_mask, M, (RES, RES))
                    
                    # Validate result
                    if np.mean(warped_img) < 15:
                        logger.warning(f"❌ Warped result too dark for {Path(img_path).name}")
                        warped_img = cv2.resize(img, (RES, RES))
                        warped_mask = cv2.resize(face_mask, (RES, RES))
                        logger.info(f"🔄 Using resize fallback - dark result")
                    else:
                        logger.info(f"✅ Successfully aligned to standard template")
            
            # Color normalization for consistent skin tones
            logger.info("Normalizing skin tone...")
            warped_normalized = normalize_skin_tone(warped_img, warped_mask)
            
            # Feature-specific weighting system
            visible_ratio = np.sum(warped_mask > 0) / (RES * RES)
            
            # Create feature-specific weight maps
            feature_weights = create_feature_weight_map(warped_mask, landmarks, M)
            
            # Base weight from quality and visible area
            base_weight = quality_score * visible_ratio
            
            # Boost weight for images with unique visible regions
            novelty_bonus = 1.0
            if successful_alignments > 0:
                # Calculate overlap with existing composite
                existing_mask = (weight_accum > 0).astype(np.float32)
                new_mask = (warped_mask > 0).astype(np.float32)
                overlap = np.sum(existing_mask * new_mask) / max(np.sum(new_mask), 1)
                novelty_bonus = 1.0 + (1.0 - overlap) * 0.6  # Higher bonus for unique regions
            
            final_weight = base_weight * novelty_bonus
            
            # Multi-scale blending for better detail preservation
            logger.info("Applying multi-scale blending...")
            
            # Create pyramid of masks for different detail levels
            mask_blur_light = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (5, 5), 1.0)
            mask_blur_medium = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (11, 11), 2.5)
            mask_blur_heavy = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (21, 21), 5.0)
            
            # Weight maps for different scales
            detail_weight = mask_blur_light * final_weight * feature_weights
            medium_weight = mask_blur_medium * final_weight * 0.8
            base_weight_map = mask_blur_heavy * final_weight * 0.6
            
            # Accumulate at different scales
            accum += warped_normalized.astype(np.float32) * detail_weight[..., np.newaxis] * 0.6
            accum += warped_normalized.astype(np.float32) * medium_weight[..., np.newaxis] * 0.3  
            accum += warped_normalized.astype(np.float32) * base_weight_map[..., np.newaxis] * 0.1
            
            weight_accum += detail_weight * 0.6 + medium_weight * 0.3 + base_weight_map * 0.1
            
            successful_alignments += 1
            logger.info(f"✅ Added to composite:")
            logger.info(f"    Base weight: {base_weight:.3f}, Novelty bonus: {novelty_bonus:.2f}")
            logger.info(f"    Final weight: {final_weight:.3f}, Brightness: {np.mean(warped_normalized):.1f}")
            logger.info(f"    Feature weights: {np.mean(feature_weights):.2f}, Multi-scale blending applied")
            
            # Save debug warped image
            debug_path = DEBUG_DIR / f"{Path(img_path).stem}_warped.jpg"
            cv2.imwrite(str(debug_path), warped_img)
            
        except Exception as e:
            logger.error(f"❌ Error aligning {Path(img_path).name}: {e}")
    
    logger.info(f"\n✅ Successfully aligned {successful_alignments} images")
    
    # Generate final composite with advanced blending
    weight_accum[weight_accum == 0] = 1e-6  # Avoid division by zero
    composite = (accum / weight_accum[..., np.newaxis]).astype(np.uint8)
    
    logger.info("Applying advanced post-processing...")
    
    # Multi-stage refinement
    composite = apply_advanced_post_processing(composite, weight_accum)
    
    # Save results with generation number and latest copy
    output_path = OUT_DIR / f"generation_{generation}_reconstruction.png"
    latest_path = OUT_DIR / "latest_reconstruction.png"
    
    cv2.imwrite(str(output_path), composite)
    cv2.imwrite(str(latest_path), composite)
    
    logger.info(f"\n🎉 RECONSTRUCTION COMPLETE!")
    logger.info(f"📸 Generation {generation} saved: {output_path}")
    logger.info(f"📸 Latest copy updated: {latest_path}")
    logger.info(f"📊 Used {successful_alignments} images")
    logger.info(f"🔍 Debug files in: {DEBUG_DIR}")
    logger.info(f"📋 Processing stats: generation_{generation}_stats.json")
    
    # List previous generations for reference
    prev_generations = sorted(OUT_DIR.glob("generation_*_reconstruction.png"))
    if len(prev_generations) > 1:
        logger.info(f"📈 Progress tracker: {len(prev_generations)} generations saved")
        for gen_file in prev_generations[-3:]:  # Show last 3
            logger.info(f"   {gen_file.name}")

if __name__ == "__main__":
    main()