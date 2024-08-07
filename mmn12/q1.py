"""
q1.py - contains the solution for question 1 of mmn 12 - Image Processing 2024c.

The scripts taken in a gray-scaled image, applies the Laplacian filter to the image
and then plots and saves the output.

Run this script with the following command:
```bash
python q1.py -h
```

The script uses the following libraries:
    - cv2
    - numpy
    - matplotlib

Please make sure to install the libraries before running the script.    

@Author: Ofir Paz
@Version: 03.08.2024
"""

# ================================== Imports ================================= #
import os
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# ============================== End Of Imports ============================== #


# ================================= Functions ================================ #
def init_argparse() -> argparse.ArgumentParser:

    """
    Initialize the argument parser for the script.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description="Question 3 - Image Processing 2024c")
    parser.add_argument("--file_name", "-f", type=str, 
                        default=Path(os.path.dirname(__file__), r"images\lena.png"), 
                        help="The relative or absolute path of the image file to process. "
                             "Default: images\\lena.png")
    parser.add_argument("--multiplier", "-c", type=float, default=-1,
                        help="The constant to multiply the Laplacian filter by. Default is -1.")
    return parser


def show_comparison(before_image: np.ndarray, after_image: np.ndarray) -> None:
    """
    Show a comparison between two images.

    Args:
        before_image (np.ndarray): The image before processing.
        after_image (np.ndarray): The image after processing.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(before_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(after_image, cmap="gray")
    plt.title("Sharpened Image")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def apply_laplacian_filter(image: np.ndarray, c: float = -1) -> np.ndarray:
    """
    Apply the Laplacian filter in the frequency domain to the image.

    Args:
        image (np.ndarray): The image to process.
        c (float): The constant to multiply the Laplacian filter by.

    Returns:
        np.ndarray: The image after applying the Laplacian filter.
    """
    P, Q = image.shape
    image = image / 255.0  # Normalize the image to [0, 1].

    # D_uv is the distance function, Eq. (4-112)
    D_uv = np.sqrt(np.square(np.arange(P) - P / 2)[:, None] + np.square(np.arange(Q) - Q / 2)[None, :])
    
    F_uv = np.fft.fft2(image)
    H_uv = -4 * np.square(np.pi) * np.square(D_uv)  # Eq. (4-124)
    laplacian = np.fft.ifft2(H_uv * F_uv)  # Eq. (4-125)
    laplacian = np.real(laplacian)  # Take the real part of the Laplacian filter.
    laplacian /= np.max(np.abs(laplacian))  # Normalize the Laplacian filter to [-1, 1].
    sharpened_image = image + c * laplacian  # Eq. (4-126)

    # Make the image positive
    sharpened_image = sharpened_image - min(np.min(sharpened_image), 0)

    # Normalize to [0, 255]
    sharpened_image = (225 * (sharpened_image / np.max(sharpened_image))).astype(np.uint8)
    
    return sharpened_image
# ============================= End Of Functions ============================= #


# =================================== Main =================================== #
def main(args: argparse.Namespace) -> None:
    """
    The main function of the script.

    Args:
        args (argparse.Namespace): The arguments for the script.

    Returns:
        None
    """
    image = cv.imread(os.path.abspath(args.file_name), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not find the image file: {os.path.abspath(args.file_name)}")
    
    sharpened_image = apply_laplacian_filter(image, c=args.multiplier)
    show_comparison(image, sharpened_image)

    image_folder = Path(args.file_name).parent
    image_name = os.path.splitext(os.path.basename(args.file_name))[0]
    cv.imwrite(str(image_folder / rf"{image_name}_sharpened_image.png"), sharpened_image)


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)
# ================================ End Of Main ================================ #