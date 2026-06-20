import os
import csv
import sys
import requests
import io
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# --- 1. POWERFUL AI ENGINE ---
import cv2

try:
    from rembg import remove, new_session
    # We use 'u2net_human_seg' -> Specifically trained for human bodies/hair
    REMBG_SESSION = new_session("u2net_human_seg") 
    REMBG_AVAILABLE = True
    print(">> AI ENGINE: HIGH (Human Segmentation Enabled)")
except ImportError:
    REMBG_AVAILABLE = False
    print(">> AI ENGINE: LOW (Background Removal unavailable)")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = 'links.csv'

# Processing Pipeline Folders
FOLDER_1_RAW   = '1_raw_downloads'
FOLDER_2_CROP  = '2_professional_crops'
FOLDER_3_FINAL = '3_final_white_bg'
FOLDER_4_JPEGS = '4_compressed_output'

# Passport Standard Dimensions
FINAL_W, FINAL_H = 413, 531
TARGET_RATIO = FINAL_W / FINAL_H
MAX_FILE_SIZE_KB = 150

# --- COMPOSITION RULES (THE "POWERFUL" PART) ---
# How much space should the face take? (Lower = Bigger Face)
# 1.8 means the crop width is 1.8x the face width.
FACE_COVERAGE_FACTOR = 1.8 

# Where should the eyes be? 
# 0.45 means the center of the face is at 45% from the top.
# This leaves room for the chest and puts eyes at the "Professional Line".
VERTICAL_OFFSET = 0.45 

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def sanitize(name):
    return "".join([c if c.isalnum() else "_" for c in name])

def get_drive_id(url):
    if 'file/d/' in url: return url.split('file/d/')[-1].split('/')[0]
    if 'id=' in url: return url.split('id=')[-1].split('&')[0]
    return None

def load_image_corrected(path):
    """Loads image and fixes phone rotation (EXIF) instantly."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except:
        return None

# ==============================================================================
# STEP 1: DOWNLOAD
# ==============================================================================
def step_1_download(data):
    print("\n[Step 1] Downloading...")
    session = requests.Session()
    
    for reg, url in tqdm(data, ascii=True):
        reg = sanitize(reg)
        path = os.path.join(FOLDER_1_RAW, f"{reg}.jpg")
        
        # Skip if already exists
        if os.path.exists(path) and os.path.getsize(path) > 0: continue
        
        fid = get_drive_id(url)
        if not fid: continue
        
        try:
            U = "https://docs.google.com/uc?export=download"
            r = session.get(U, params={'id': fid}, stream=True)
            if 'download_warning' in r.text:
                tok = r.cookies.get_dict().get('download_warning')
                if tok: r = session.get(U, params={'id': fid, 'confirm': tok}, stream=True)
            
            if r.status_code == 200:
                with open(path, "wb") as f: f.write(r.content)
        except: pass

# ==============================================================================
# STEP 2: PROFESSIONAL COMPOSITION CROP
# ==============================================================================
def step_2_smart_crop(data):
    print("\n[Step 2] Applying Professional Composition (OpenCV DNN)...")
    
    # Load OpenCV DNN face detector (more compatible with Python 3.14)
    face_net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "opencv_face_detector.caffemodel")
    
    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        in_path = os.path.join(FOLDER_1_RAW, f"{reg}.jpg")
        out_path = os.path.join(FOLDER_2_CROP, f"{reg}.png") # PNG to keep quality before final

        if not os.path.exists(in_path): continue
        if os.path.exists(out_path): continue

        img = load_image_corrected(in_path)
        if img is None: continue

        # Detect Face using OpenCV DNN
        w_img, h_img = img.size
        np_img = np.array(img)
        
        # Create blob from image (OpenCV requires specific preprocessing)
        blob = cv2.dnn.blobFromImage(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR), 1.0, (300, 300), [104.0, 177.0, 123.0])
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Parse detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # min_detection_confidence
                box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                faces.append((confidence, box))
        
        if not faces:
            # Fallback: If no face found, just resize the center
            img.resize((FINAL_W, FINAL_H)).save(out_path)
            continue

        # 1. Get the Detection Data (use highest confidence face)
        best_face = max(faces, key=lambda x: x[0])
        box = best_face[1]
        
        # Face pixel coordinates
        fx, fy, fx2, fy2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        fw = fx2 - fx
        fh = fy2 - fy

        # 2. Calculate the "Perfect Center"
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2

        # 3. Calculate Crop Dimensions (The Passport Ratio)
        # We base the crop size on the face width to ensure consistency
        crop_width = int(fw * FACE_COVERAGE_FACTOR)
        crop_height = int(crop_width / TARGET_RATIO)

        # 4. Determine Coordinates
        # We shift the box UP so the face isn't dead center (looks amateur). 
        # We want the face slightly higher (Professional Standard).
        x1 = face_center_x - (crop_width // 2)
        y1 = face_center_y - int(crop_height * VERTICAL_OFFSET)
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # 5. THE "INFINITE CANVAS" TECHNIQUE
        # This prevents black bars if the person is at the edge of the photo.
        # We create a massive white background and paste the photo in the middle.
        
        # Canvas size: 3x the original image
        canvas_w = max(w_img * 3, 3000)
        canvas_h = max(h_img * 3, 3000)
        
        infinite_canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        
        # Calculate offset to center the original image on the canvas
        offset_x = (canvas_w - w_img) // 2
        offset_y = (canvas_h - h_img) // 2
        
        infinite_canvas.paste(img, (offset_x, offset_y))

        # Adjust crop coordinates to the new canvas system
        final_x1 = x1 + offset_x
        final_y1 = y1 + offset_y
        final_x2 = x2 + offset_x
        final_y2 = y2 + offset_y

        # Perform the Crop
        crop_img = infinite_canvas.crop((final_x1, final_y1, final_x2, final_y2))
        
        # High Quality Resize
        crop_img = crop_img.resize((FINAL_W, FINAL_H), Image.Resampling.LANCZOS)
        crop_img.save(out_path)

# ==============================================================================
# STEP 3: BACKGROUND REMOVAL (ON THE PERFECT CROP)
# ==============================================================================
def step_3_white_background(data):
    print("\n[Step 3] AI Background Removal (To White)...")
    if not REMBG_AVAILABLE: return

    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        in_path = os.path.join(FOLDER_2_CROP, f"{reg}.png")
        out_path = os.path.join(FOLDER_3_FINAL, f"{reg}.png")

        if not os.path.exists(in_path): continue
        if os.path.exists(out_path): continue

        try:
            # Load
            img = Image.open(in_path).convert("RGBA")
            
            # AI Remove (alpha_matting=True makes hair edges soft, not jagged)
            cutout = remove(img, session=REMBG_SESSION, alpha_matting=True, alpha_matting_foreground_threshold=240)
            
            # Composite onto pure white
            white_bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
            final_result = Image.alpha_composite(white_bg, cutout)
            
            # Save
            final_result.save(out_path)
            
        except Exception:
            # If AI fails (rare), we copy the crop so we at least have the photo
            try: Image.open(in_path).save(out_path)
            except: pass

# ==============================================================================
# STEP 4: COMPRESSION & FORMATTING
# ==============================================================================
def step_4_compress(data):
    print(f"\n[Step 4] Final Compression (<{MAX_FILE_SIZE_KB}KB)...")
    count = 0
    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        # Priority: Processed BG -> Cropped BG -> Raw
        in_path = os.path.join(FOLDER_3_FINAL, f"{reg}.png")
        if not os.path.exists(in_path):
             in_path = os.path.join(FOLDER_2_CROP, f"{reg}.png")
        
        out_path = os.path.join(FOLDER_4_JPEGS, f"{reg}.jpg")

        if not os.path.exists(in_path): continue
        # if os.path.exists(out_path): count += 1; continue # Unwrap to force re-compress if needed

        try:
            img = Image.open(in_path).convert("RGB")
            
            # Smart Compression Loop
            quality = 95
            while quality > 10:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                size_kb = buf.tell() / 1024
                
                if size_kb < MAX_FILE_SIZE_KB:
                    with open(out_path, "wb") as f: f.write(buf.getvalue())
                    count += 1
                    break
                quality -= 5
        except: pass
    return count

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    # Create Directory Structure
    folders = [FOLDER_1_RAW, FOLDER_2_CROP, FOLDER_3_FINAL, FOLDER_4_JPEGS]
    for f in folders:
        os.makedirs(f, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' file missing.")
        sys.exit()

    # Read Data
    with open(INPUT_CSV, 'r') as f:
        csv_data = [row for row in csv.reader(f) if row and len(row) >= 2]
    
    print(f"--- PROCESSING {len(csv_data)} PHOTOS ---")
    
    # Run Pipeline
    step_1_download(csv_data)
    step_2_smart_crop(csv_data)
    step_3_white_background(csv_data)
    total_done = step_4_compress(csv_data)
    
    print("="*40)
    print(f"COMPLETE! {total_done} valid passport photos generated.")
    print(f"Output folder: {FOLDER_4_JPEGS}")
    print("="*40)
    input("Press Enter to exit...")