import subprocess
import sys

def install_packages():
    packages = [
        "requests",
        "tqdm",
        "opencv-python",
        "Pillow",
        "numpy",
        "scipy",
        "rembg"
    ]
    
    print("Starting package installation...")
    print(f"Using Python interpreter: {sys.executable}\n")
    
    # First upgrade pip
    try:
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("pip upgraded successfully.\n")
    except Exception as e:
        print(f"Warning: Failed to upgrade pip: {e}\n")
        
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            # Running with -m pip install ensures we install on the correct active environment
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to install {package}. Exit code: {e.returncode}\n")
            failed_packages.append(package)
            
    if failed_packages:
        print("--- Installation Summary ---")
        print(f"Failed to install the following packages: {', '.join(failed_packages)}")
        print("Please check your internet connection or install permissions.")
        sys.exit(1)
    else:
        print("--- Installation Summary ---")
        print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages()
