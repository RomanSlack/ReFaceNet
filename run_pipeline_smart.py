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

def analyze_image_quality(img):
    """Analyze image for face reconstruction quality"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    quality_score = (laplacian_var / 100) + (contrast / 50) + (1 - abs(brightness - 127) / 127)
    
    logger.info(f"   Sharpness: {laplacian_var:.1f}, Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
    logger.info(f"   Quality Score: {quality_score:.2f}")
    
    return quality_score

def detect_face_with_logging(img, img_path):
    """Detect face with detailed logging"""
    logger.info(f"\n=== Processing: {Path(img_path).name} ===")
    logger.info(f"Image shape: {img.shape}")
    
    h, w = img.shape[:2]
    
    # Analyze image quality
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
        
        # Create face mask
        face_mask = create_face_mask(img, landmarks)
        visible_area = np.sum(face_mask > 0)
        
        logger.info(f"Visible face area: {visible_area} pixels")
        
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
    
    # Better landmark selection for stable face alignment
    # Eyes: inner and outer corners
    left_eye = [133, 33]    # left eye outer, inner corner
    right_eye = [362, 263]  # right eye inner, outer corner  
    nose_tip = [1]          # nose tip
    mouth = [61, 291]       # mouth corners
    
    key_indices = left_eye + right_eye + nose_tip + mouth
    template_key_points = template_landmarks[key_indices]
    
    logger.info(f"Template key points shape: {template_key_points.shape}")
    logger.info(f"Eye distance: {np.linalg.norm(template_key_points[0] - template_key_points[2]):.1f}px")
    
    # Initialize accumulation
    accum = np.zeros((RES, RES, 3), np.float32)
    weight_accum = np.zeros((RES, RES), np.float32)
    
    successful_alignments = 0
    
    for i, (img_path, img, landmarks, face_mask, quality_score) in enumerate(valid_images):
        logger.info(f"\nAligning image {i+1}: {Path(img_path).name}")
        
        try:
            # Align using key landmarks
            src_points = landmarks[key_indices]
            
            # Validate points before alignment
            if len(src_points) < 4:
                logger.warning(f"❌ Not enough key points for {Path(img_path).name}")
                continue
                
            # Check for valid point spread
            point_spread = np.std(src_points, axis=0)
            if np.any(point_spread < 20):  # Increased threshold for better stability
                logger.warning(f"❌ Points too close together for {Path(img_path).name}")
                warped_img = cv2.resize(img, (RES, RES))
                warped_mask = cv2.resize(face_mask, (RES, RES))
                logger.info(f"🔄 Using resize fallback due to point spread")
            else:
                # Calculate eye distance for this image to check if reasonable
                eye_dist_src = np.linalg.norm(src_points[0] - src_points[2])
                eye_dist_template = np.linalg.norm(template_key_points[0] - template_key_points[2])
                scale_ratio = eye_dist_src / eye_dist_template
                
                logger.info(f"Eye distance ratio: {scale_ratio:.2f}")
                
                # Only use transformation if scale ratio is reasonable
                if scale_ratio < 0.3 or scale_ratio > 3.0:
                    logger.warning(f"❌ Unreasonable scale ratio for {Path(img_path).name}")
                    warped_img = cv2.resize(img, (RES, RES))
                    warped_mask = cv2.resize(face_mask, (RES, RES))
                    logger.info(f"🔄 Using resize fallback due to scale ratio")
                else:
                    # Try similarity transform (only scale, rotation, translation - no shear)
                    M, _ = cv2.estimateAffinePartial2D(src_points, template_key_points)
                    
                    if M is None:
                        logger.warning(f"❌ Failed to compute alignment for {Path(img_path).name}")
                        warped_img = cv2.resize(img, (RES, RES))
                        warped_mask = cv2.resize(face_mask, (RES, RES))
                        logger.info(f"🔄 Using resize fallback - no matrix")
                    else:
                        # Validate transformation matrix
                        det = np.linalg.det(M[:2, :2])
                        if abs(det) < 0.3 or abs(det) > 3.0:  # Stricter bounds
                            logger.warning(f"❌ Invalid transformation matrix det={det:.2f} for {Path(img_path).name}")
                            warped_img = cv2.resize(img, (RES, RES))
                            warped_mask = cv2.resize(face_mask, (RES, RES))
                            logger.info(f"🔄 Using resize fallback - bad matrix")
                        else:
                            # Warp image and mask
                            warped_img = cv2.warpAffine(img, M, (RES, RES))
                            warped_mask = cv2.warpAffine(face_mask, M, (RES, RES))
                            
                            # Check if warp result is valid (not black)
                            if np.mean(warped_img) < 20:
                                logger.warning(f"❌ Warped image too dark for {Path(img_path).name}")
                                warped_img = cv2.resize(img, (RES, RES))
                                warped_mask = cv2.resize(face_mask, (RES, RES))
                                logger.info(f"🔄 Using resize fallback - dark result")
                            else:
                                logger.info(f"✅ Successfully warped {Path(img_path).name}")
            
            # Weight by quality and visible area
            weight = quality_score * (np.sum(warped_mask > 0) / (RES * RES))
            
            # Accumulate with weights
            mask_norm = warped_mask.astype(np.float32) / 255.0
            weight_map = mask_norm * weight
            
            accum += warped_img.astype(np.float32) * weight_map[..., np.newaxis]
            weight_accum += weight_map
            
            successful_alignments += 1
            logger.info(f"✅ Added to composite with weight: {weight:.3f}, mean brightness: {np.mean(warped_img):.1f}")
            
            # Save debug warped image
            debug_path = DEBUG_DIR / f"{Path(img_path).stem}_warped.jpg"
            cv2.imwrite(str(debug_path), warped_img)
            
        except Exception as e:
            logger.error(f"❌ Error aligning {Path(img_path).name}: {e}")
    
    logger.info(f"\n✅ Successfully aligned {successful_alignments} images")
    
    # Generate final composite
    weight_accum[weight_accum == 0] = 1e-6  # Avoid division by zero
    composite = (accum / weight_accum[..., np.newaxis]).astype(np.uint8)
    
    # Save results with generation number
    output_path = OUT_DIR / f"generation_{generation}_reconstruction.png"
    cv2.imwrite(str(output_path), composite)
    
    # Also save as latest for convenience
    latest_path = OUT_DIR / "latest_reconstruction.png"
    cv2.imwrite(str(latest_path), composite)
    
    logger.info(f"\n🎉 RECONSTRUCTION COMPLETE!")
    logger.info(f"📸 Output saved: {output_path}")
    logger.info(f"📸 Latest copy: {latest_path}")
    logger.info(f"📊 Used {successful_alignments} images")
    logger.info(f"🔍 Debug files in: {DEBUG_DIR}")
    logger.info(f"📋 Processing stats: {stats_file}")

if __name__ == "__main__":
    main()