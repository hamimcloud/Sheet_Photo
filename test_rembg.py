try:
    import numpy
    import scipy
    from rembg import remove
    print(f"Numpy version: {numpy.__version__}")
    print(f"Scipy version: {scipy.__version__}")
    print("SUCCESS: rembg is working now.")
except Exception as e:
    print(f"Still failing: {e}")