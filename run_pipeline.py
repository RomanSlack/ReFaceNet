import cv2, numpy as np, mediapipe as mp, glob, torch
from tqdm import tqdm
from pathlib import Path

RAW_DIR = Path("data/raw")          # your face photos
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)
RES     = 512                       # canonical face chip size

# Initialize face detection with lower confidence for better detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.3)

# ---------- helper functions ----------------
def detect_largest_face(img):
    """Detect the largest face in image using MediaPipe"""
    h, w = img.shape[:2]
    results = mp_face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        return None
    
    # Find largest face
    largest_area = 0
    best_bbox = None
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        area = bbox.width * bbox.height
        if area > largest_area:
            largest_area = area
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            best_bbox = (x, y, width, height)
    
    return best_bbox

def landmarks_norm(img, face_bbox=None):
    """Extract facial landmarks with optional face region"""
    h, w = img.shape[:2]
    
    # Crop to face region if provided
    if face_bbox:
        x, y, width, height = face_bbox
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        width = min(w - x, width + 2*pad)
        height = min(h - y, height + 2*pad)
        face_img = img[y:y+height, x:x+width]
        offset = (x, y)
    else:
        face_img = img
        offset = (0, 0)
    
    fh, fw = face_img.shape[:2]
    res = mp_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    
    if not res.multi_face_landmarks:
        return None
    
    pts = np.array([[lm.x*fw + offset[0], lm.y*fh + offset[1]] 
                   for lm in res.multi_face_landmarks[0].landmark])
    return pts

# Process images with better face detection
print("Processing images with improved face detection...")
valid_images = []
all_landmarks = []

for p in tqdm(sorted(glob.glob(str(RAW_DIR/"*")))):
    img = cv2.imread(p)
    if img is None:
        continue
    
    # Detect face first
    face_bbox = detect_largest_face(img)
    
    # Get landmarks
    pts = landmarks_norm(img, face_bbox)
    
    if pts is not None:
        valid_images.append((p, img, pts))
        all_landmarks.append(pts)
        print(f"✓ {Path(p).name}: Found face and landmarks")
    else:
        print(f"✗ {Path(p).name}: No landmarks detected")

if len(valid_images) < 2:
    print(f"❌ Only found {len(valid_images)} valid faces. Need at least 2.")
    exit(1)

print(f"✅ Processing {len(valid_images)} valid images")

# Use first valid image as template
tpl_pts = all_landmarks[0]
sel_idx = [33,133,263,362,1,61,291,199]  # stable landmarks (eyes, nose, mouth)
tpl_sub = tpl_pts[sel_idx]

accum = np.zeros((RES,RES,3), np.float32)
seen = np.zeros((RES,RES), np.uint16)

print("Aligning and compositing faces...")
successful_alignments = 0

for p, img, pts in tqdm(valid_images):
    try:
        M, _ = cv2.estimateAffinePartial2D(pts[sel_idx], tpl_sub)
        if M is not None:
            warped = cv2.warpAffine(img, M, (RES,RES))
            mask = cv2.warpAffine(np.ones(img.shape[:2], np.uint8), M, (RES,RES))
            
            accum += warped.astype(np.float32)
            seen += mask
            successful_alignments += 1
        else:
            print(f"⚠️ Failed to align {Path(p).name}")
    except Exception as e:
        print(f"✗ Error processing {Path(p).name}: {e}")

print(f"✅ Successfully aligned {successful_alignments} images")

# Generate composite
seen[seen==0] = 1
composite = (accum/seen[...,None]).astype(np.uint8)
cv2.imwrite(str(OUT_DIR/"composite_improved.png"), composite)

print(f"✅ Face reconstruction completed → outputs/composite_improved.png")
print(f"   Used {successful_alignments}/{len(valid_images)} images")
