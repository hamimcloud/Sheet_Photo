# File: check_environment.py
import sys
import os

print("--- PYTHON ENVIRONMENT CHECK ---")

print(f"\n[1] Python Executable Path:")
print(f"    This script is being run by: {sys.executable}")

print(f"\n[2] Python Version:")
print(f"    Version: {sys.version}")

print("\n[3] System Path (where Python looks for files):")
for path in sys.path:
    print(f"    - {path}")

print("\n[4] Checking for required libraries...")
try:
    import requests
    print("    SUCCESS: 'requests' is installed.")
except ImportError:
    print("    FAIL: 'requests' is NOT installed.")

try:
    import cv2
    print("    SUCCESS: 'opencv-python' (cv2) is installed.")
except ImportError:
    print("    FAIL: 'opencv-python' (cv2) is NOT installed.")
    
try:
    import PIL
    print("    SUCCESS: 'Pillow' (PIL) is installed.")
except ImportError:
    print("    FAIL: 'Pillow' (PIL) is NOT installed.")

try:
    import rembg
    print("    SUCCESS: 'rembg' is installed.")
except ImportError:
    print("    FAIL: 'rembg' is NOT installed.")

print("\n--- CHECK COMPLETE ---")