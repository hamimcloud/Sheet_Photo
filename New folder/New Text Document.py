import os
import csv
import requests
import cv2
from PIL import Image
from tqdm import tqdm

# --- Main Configuration ---
INPUT_CSV_FILE = 'links.csv'
DOWNLOAD_FOLDER = 'downloaded_images'
CROPPED_FOLDER = 'cropped_passport_photos'
HAAR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'

# --- Cropping Parameters ---
# Desired final size in pixels (Width, Height). Common passport sizes:
# - USA (2x2 inch @ 300dpi): (600, 600)
# - Europe/India (35x45 mm @ 300dpi): (413, 531)
FINAL_SIZE = (413, 531)
TARGET_ASPECT_RATIO = FINAL_SIZE[0] / FINAL_SIZE[1]
# Padding around the detected face to capture head and shoulders
PADDING_TOP = 0.60
PADDING_SIDES = 0.50

# --- Helper Function: Sanitize Filename ---
def sanitize_filename(name, extension):
    """Replaces invalid characters for filenames and appends an extension."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return f"{name}.{extension}"

# ==============================================================================
# SECTION 1: GOOGLE DRIVE DOWNLOADING LOGIC
# ==============================================================================

def get_google_drive_file_id(url):
    """Extracts the file ID from various Google Drive URL formats."""
    parts = url.split('/')
    for i, part in enumerate(parts):
        if part == 'd':
            return parts[i + 1]
    if 'id=' in url:
        return url.split('id=')[-1].split('&')[0]
    return None

def download_file_from_google_drive(file_id, destination):
    """Downloads a file from Google Drive, handling virus scan warnings."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    try:
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"  -> Network error for {destination}: {e}")
        return False

# ==============================================================================
# SECTION 2: FACE DETECTION AND CROPPING LOGIC
# ==============================================================================

def download_haar_cascade():
    """Downloads the OpenCV face detection model if it's missing."""
    URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"Downloading face detection model ('{HAAR_CASCADE_FILE}')...")
        try:
            r = requests.get(URL, allow_redirects=True)
            r.raise_for_status()
            with open(HAAR_CASCADE_FILE, 'wb') as f:
                f.write(r.content)
            print("Download complete.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"\nFATAL ERROR: Could not download the face detection model. Check your internet connection. Error: {e}")
            return False
    return True

def crop_and_resize_image(image_path, face_cascade):
    """Detects a face, crops it to passport size, and returns the cropped image."""
    try:
        image_cv = cv2.imread(image_path)
        if image_cv is None: return None
        
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0: return None
        
        main_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = main_face

        face_center_x = x + w // 2
        new_w = int(w * (1 + 2 * PADDING_SIDES))
        new_h = int(new_w / TARGET_ASPECT_RATIO)
        new_x = max(0, face_center_x - (new_w // 2))
        new_y = max(0, y - int(h * PADDING_TOP))

        img_h, img_w, _ = image_cv.shape
        if new_x + new_w > img_w: new_w = img_w - new_x
        if new_y + new_h > img_h: new_h = img_h - new_y
            
        img_pil = Image.open(image_path)
        cropped_img = img_pil.crop((new_x, new_y, new_x + new_w, new_y + new_h))
        resized_img = cropped_img.resize(FINAL_SIZE, Image.Resampling.LANCZOS)
        return resized_img

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

# ==============================================================================
# SECTION 3: MAIN EXECUTION SCRIPT
# ==============================================================================

def main():
    """Main function to run the full download and crop process."""
    print("--- Image Downloader and Passport Photo Cropper ---")

    # --- Initial Checks ---
    if not download_haar_cascade(): return
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"\nERROR: Input file '{INPUT_CSV_FILE}' not found! Please create it.")
        return
        
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(CROPPED_FOLDER, exist_ok=True)
    
    # --- PART 1: DOWNLOAD IMAGES ---
    print(f"\n[PART 1/2] Reading '{INPUT_CSV_FILE}' and downloading images...")
    try:
        with open(INPUT_CSV_FILE, 'r', newline='') as f:
            download_list = [row for row in csv.reader(f) if row and len(row) == 2]
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not download_list:
        print("No valid entries found in the CSV file.")
        return
        
    download_success_count = 0
    successfully_downloaded = []
    
    for reg_number, url in tqdm(download_list, desc="Downloading"):
        file_id = get_google_drive_file_id(url.strip())
        if not file_id:
            print(f"Warning: Could not get a file ID from URL: {url}")
            continue

        filename = sanitize_filename(reg_number.strip(), 'png')
        dest_path = os.path.join(DOWNLOAD_FOLDER, filename)
        
        if download_file_from_google_drive(file_id, dest_path):
            download_success_count += 1
            successfully_downloaded.append(dest_path)
    
    print(f"Download complete. {download_success_count}/{len(download_list)} images downloaded.")

    # --- PART 2: CROP IMAGES ---
    print(f"\n[PART 2/2] Detecting faces and cropping images...")
    if not successfully_downloaded:
        print("No images were downloaded, so skipping cropping.")
        return
        
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    crop_success_count = 0

    for image_path in tqdm(successfully_downloaded, desc="Cropping   "):
        final_image = crop_and_resize_image(image_path, face_cascade)
        if final_image:
            base_filename = os.path.basename(image_path)
            output_path = os.path.join(CROPPED_FOLDER, base_filename)
            final_image.save(output_path, "PNG")
            crop_success_count += 1

    # --- FINAL REPORT ---
    print("\n--- Processing Complete ---")
    print(f"Images Downloaded: {download_success_count}")
    print(f"Images Cropped:    {crop_success_count} (out of {len(successfully_downloaded)} downloaded)")
    print("-" * 27)
    print(f"Original images are in: '{DOWNLOAD_FOLDER}'")
    print(f"Final passport photos are in: '{CROPPED_FOLDER}'")

if __name__ == "__main__":
    main()