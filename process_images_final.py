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
FOLDER_1_RAW       = '1_raw_downloads'
FOLDER_2_WHITE_BG  = '2_white_bg'
FOLDER_3_FINAL     = '3_final_white_bg'
FOLDER_4_JPEGS     = '4_compressed_output'

# Passport Standard Dimensions
FINAL_W, FINAL_H = 413, 531
TARGET_RATIO = FINAL_W / FINAL_H
MAX_FILE_SIZE_KB = 150

# --- COMPOSITION RULES (THE "POWERFUL" PART) ---
# How much space should the face take? (Lower = Bigger Face)
# 2.0 provides better margin to avoid cutting hair or shoulders.
FACE_COVERAGE_FACTOR = 2.0

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

def center_crop_to_ratio(img, target_ratio):
    """Crops the center of the image to match the target aspect ratio without distortion."""
    w, h = img.size
    current_ratio = w / h
    if current_ratio > target_ratio:
        # Image is too wide - crop left and right margins
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        y1 = 0
        x2 = x1 + new_w
        y2 = h
    else:
        # Image is too tall - crop top and bottom margins (leaving slightly more room at the top)
        new_h = int(w / target_ratio)
        x1 = 0
        y1 = int((h - new_h) * 0.4)  # Shift crop slightly up (0.4 instead of 0.5) to keep head in frame
        x2 = w
        y2 = y1 + new_h
    return img.crop((x1, y1, x2, y2))

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
# STEP 2: AI BACKGROUND REMOVAL (ON THE FULL IMAGE)
# ==============================================================================
def step_2_remove_background(data):
    print("\n[Step 2] AI Background Removal (Full Image to White)...")
    if not REMBG_AVAILABLE:
        print(">> Skipping Step 2: rembg is not available.")
        return

    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        in_path = os.path.join(FOLDER_1_RAW, f"{reg}.jpg")
        out_path = os.path.join(FOLDER_2_WHITE_BG, f"{reg}.png")

        if not os.path.exists(in_path): continue
        if os.path.exists(out_path): continue

        try:
            # Load and fix EXIF orientation
            img = load_image_corrected(in_path)
            if img is None: continue

            # Downscale large images to max 1600px dimension to optimize processing speed and clarity
            max_dim = 1600
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            # Convert to RGBA for alpha matting
            img_rgba = img.convert("RGBA")

            # Run Background Removal (we set alpha_matting=False to prevent background color bleed/halos)
            cutout = remove(
                img_rgba, 
                session=REMBG_SESSION, 
                alpha_matting=False, 
                post_process_mask=True
            )

            # Composite onto pure white background
            white_bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
            final_result = Image.alpha_composite(white_bg, cutout).convert("RGB")

            # Save full image with white background
            final_result.save(out_path)

        except Exception as e:
            # Fallback: Copy raw image if AI fails
            try:
                img = load_image_corrected(in_path)
                if img:
                    img.save(out_path)
            except:
                pass

# ==============================================================================
# STEP 3: PROFESSIONAL COMPOSITION CROP (ON WHITE BG IMAGE)
# ==============================================================================
def step_3_smart_crop(data):
    print("\n[Step 3] Applying Professional Composition (OpenCV DNN on White BG)...")
    
    # Load OpenCV DNN face detector
    try:
        face_net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "opencv_face_detector.caffemodel")
    except Exception as e:
        print(f">> ERROR: Failed to load OpenCV DNN face detector: {e}")
        return

    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        
        # We read from FOLDER_2_WHITE_BG. Fallback to FOLDER_1_RAW if background removal skipped.
        in_path = os.path.join(FOLDER_2_WHITE_BG, f"{reg}.png")
        if not os.path.exists(in_path):
            in_path = os.path.join(FOLDER_1_RAW, f"{reg}.jpg")
            
        out_path = os.path.join(FOLDER_3_FINAL, f"{reg}.png")

        if not os.path.exists(in_path): continue
        if os.path.exists(out_path): continue

        img = load_image_corrected(in_path)
        if img is None: continue

        w_img, h_img = img.size
        np_img = np.array(img)
        
        # Create blob from image for OpenCV DNN (300x300 target size)
        blob = cv2.dnn.blobFromImage(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR), 1.0, (300, 300), [104.0, 177.0, 123.0])
        face_net.setInput(blob)
        detections = face_net.forward()
        
        # Parse detections with a lower confidence threshold (0.45) for better robustness
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.45:
                # Bounding box coordinates clamped to image dimensions
                box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                faces.append((confidence, box))
        
        if not faces:
            # Fallback: crop the center preserving the aspect ratio (no squishing!)
            cropped = center_crop_to_ratio(img, TARGET_RATIO)
            cropped.resize((FINAL_W, FINAL_H), Image.Resampling.LANCZOS).save(out_path)
            continue

        # Get highest confidence face detection
        best_face = max(faces, key=lambda x: x[0])
        box = best_face[1]
        
        # Face pixel coordinates (clamped to image dimensions)
        fx = max(0, min(int(box[0]), w_img))
        fy = max(0, min(int(box[1]), h_img))
        fx2 = max(0, min(int(box[2]), w_img))
        fy2 = max(0, min(int(box[3]), h_img))
        
        fw = fx2 - fx
        fh = fy2 - fy

        # Calculate face center coordinates
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2

        # Calculate crop width and height matching TARGET_RATIO
        crop_width = int(fw * FACE_COVERAGE_FACTOR)
        crop_height = int(crop_width / TARGET_RATIO)

        # Determine raw crop coordinates before padding checks
        x1 = face_center_x - (crop_width // 2)
        y1 = face_center_y - int(crop_height * VERTICAL_OFFSET)
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # THE SEAMLESS CANVAS TECHNIQUE:
        # Since the background of the source image is already pure white, we create a
        # padded canvas filled with pure white to allow cropping outside coordinates safely.
        canvas_w = max(w_img * 3, 3000)
        canvas_h = max(h_img * 3, 3000)
        
        infinite_canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        
        # Center the original white-bg image on the white canvas (fully seamless)
        offset_x = (canvas_w - w_img) // 2
        offset_y = (canvas_h - h_img) // 2
        infinite_canvas.paste(img, (offset_x, offset_y))

        # Translate coordinates to padded canvas system
        final_x1 = x1 + offset_x
        final_y1 = y1 + offset_y
        final_x2 = x2 + offset_x
        final_y2 = y2 + offset_y

        # Perform the seamless crop
        crop_img = infinite_canvas.crop((final_x1, final_y1, final_x2, final_y2))
        
        # Resize to standard passport resolution and save
        crop_img = crop_img.resize((FINAL_W, FINAL_H), Image.Resampling.LANCZOS)
        crop_img.save(out_path)

# ==============================================================================
# STEP 4: COMPRESSION & FORMATTING
# ==============================================================================
def step_4_compress(data):
    print(f"\n[Step 4] Final Compression (<{MAX_FILE_SIZE_KB}KB)...")
    count = 0
    for reg, _ in tqdm(data, ascii=True):
        reg = sanitize(reg)
        
        in_path = os.path.join(FOLDER_3_FINAL, f"{reg}.png")
        out_path = os.path.join(FOLDER_4_JPEGS, f"{reg}.jpg")

        if not os.path.exists(in_path): continue

        try:
            img = Image.open(in_path).convert("RGB")
            
            # Iteratively compress quality to hit file size target
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
        except Exception as e:
            print(f">> Failed to compress {reg}: {e}")
            
    return count

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    # Create Directory Structure
    folders = [FOLDER_1_RAW, FOLDER_2_WHITE_BG, FOLDER_3_FINAL, FOLDER_4_JPEGS]
    for f in folders:
        os.makedirs(f, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' file missing.")
        sys.exit()

    # Read Data
    with open(INPUT_CSV, 'r') as f:
        csv_data = [row for row in csv.reader(f) if row and len(row) >= 2]
    
    print(f"--- PROCESSING {len(csv_data)} PHOTOS ---")
    
    # Run Optimized Pipeline
    step_1_download(csv_data)
    step_2_remove_background(csv_data)
    step_3_smart_crop(csv_data)
    total_done = step_4_compress(csv_data)
    
    print("="*40)
    print(f"COMPLETE! {total_done} valid passport photos generated.")
    input("Press Enter to exit...")