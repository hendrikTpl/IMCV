import numpy as np
import cv2
import math
from typing import Tuple

def compute_spatial_frequency(image: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the Spatial Frequency (SF) of a grayscale image.
    
    Returns:
        RF: Row Frequency
        CF: Column Frequency
        SF: Spatial Frequency
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    M, N = image.shape
    img = image.astype(np.float32)

    # Row Frequency (RF)
    diff_rows = np.diff(img, axis=1)
    rf = np.sqrt(np.sum(diff_rows ** 2) / (M * N))
    # Column Frequency (CF)
    diff_cols = np.diff(img, axis=0)
    cf = np.sqrt(np.sum(diff_cols ** 2) / (M * N))
    # Spatial Frequency (SF)
    sf = math.sqrt(rf ** 2 + cf ** 2)
    return rf, cf, sf

if __name__ == "__main__":
    image_path = "sample.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    rf, cf, sf = compute_spatial_frequency(image)
    print(f"Row Frequency (RF): {rf:.4f}")
    print(f"Column Frequency (CF): {cf:.4f}")
    print(f"Spatial Frequency (SF): {sf:.4f}")
