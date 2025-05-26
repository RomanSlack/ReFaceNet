import cv2, numpy as np, mediapipe as mp, glob, torch
from tqdm import tqdm
from simple_lama_inpainting import SimpleLama      # LaMa wrapper
from pathlib import Path

RAW_DIR = Path("data/raw")          # your 100 cropped photos
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)
RES     = 256                       # canonical face chip size

mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
lama    = SimpleLama()              # lazy-loads weights on first call

# ---------- helper ----------------
def landmarks_norm(img):
    h, w = img.shape[:2]
    res   = mp_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return None
    pts = np.array([[lm.x*w, lm.y*h] for lm in res.multi_face_landmarks[0].landmark])
    return pts

# 1) pick first image as landmark template
tpl_img = cv2.imread(sorted(glob.glob(str(RAW_DIR/"*")))[0])
tpl_pts = landmarks_norm(tpl_img)
sel_idx = [33,133,263,362,1,61,291,199]           # stable eight-point subset (eyes, nose tip, chin)
tpl_sub = tpl_pts[sel_idx]

accum   = np.zeros((RES,RES,3), np.float32)
seen    = np.zeros((RES,RES),   np.uint16)

for p in tqdm(sorted(glob.glob(str(RAW_DIR/"*")))):
    img  = cv2.imread(p)
    pts  = landmarks_norm(img)
    if pts is None: continue
    M,_  = cv2.estimateAffinePartial2D(pts[sel_idx], tpl_sub)  # 2-D similarity
    warped = cv2.warpAffine(img, M, (RES,RES))
    mask   = cv2.warpAffine(np.ones(img.shape[:2],np.uint8), M, (RES,RES))

    accum += warped.astype(np.float32)
    seen  += mask

# 2) average visible pixels
seen[seen==0] = 1
composite = (accum/seen[...,None]).astype(np.uint8)
cv2.imwrite(str(OUT_DIR/"composite_raw.png"), composite)

# 3) build hole mask & in-paint
hole_mask = (seen==1).astype(np.uint8)*255        # pixels never observed
result = lama(composite, hole_mask)
result.save(OUT_DIR/"reconstructed_face.png")
print("✅ 2-D reconstruction done → outputs/reconstructed_face.png")
