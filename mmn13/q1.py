"""
q1.py - contains the solution for question 1 of mmn 13 - Image Processing 2024c.

The scripts taken in a gray-scaled image and applies local histogram 
equalization to the image. The image path and the window size can be given.

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
@Version: 19.08.2024
"""

# ================================== Imports ================================= #
import os
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
# ============================== End Of Imports ============================== #


# ================================= Functions ================================ #
def init_argparse() -> argparse.ArgumentParser:

    """
    Initialize the argument parser for the script.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description="Question 1 - Image Processing 2024c")
    parser.add_argument("--file_name", "-f", type=str, 
                        default=Path(os.path.dirname(__file__), r"images\embedded_squares.jpg"), 
                        help="The relative or absolute path of the image file to process. "
                             "Default: images\\embedded_squares.jpg")
    parser.add_argument("--window_size", "--ws", nargs=2, type=int, default=(3, 3), 
                        metavar=("rows", "cols"), 
                        help="The size of the window for the local histogram equalization. "
                             "Default: (3, 3)")
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
    plt.title("Transformed Image")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def get_neighborhood_value(window: np.ndarray) -> np.ndarray:
    """
    Get the neighborhood value of the image.
    
    Args:
        image (np.ndarray): The image to process.

    Returns:
        np.ndarray: The image after applying the histogram equalization.
    """
    # Extracts number of rows and cols, `M` and `N` respectively.
    M, N = window.shape

    # Set the number of levels.
    L = 256

    # Compute the normalized histogram of the window; Eq. (3-14).
    histogram = (window.reshape(1, -1) == np.arange(L - 1).reshape(-1, 1)).sum(axis=1) / (M * N)

    # Compute the transformation; Eq. (3-15).
    transformation = lambda r: round((L - 1) * histogram[:r + 1].sum())

    # Return the transformation of the center pixel.
    return transformation(window[M // 2, N // 2])


def apply_local_histogram_equlization(image: np.ndarray, window_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply the local histogram equalization to the image.

    Args:
        image (np.ndarray): The image to process (assumed dtype=np.uint8).
        window_size (Tuple[int, int]): The size of the window for the local histogram equalization.

    Returns:
        np.ndarray: The image after applying the local histogram equalization.
    """
    win_h, win_w = window_size  # Extract height and width of the window size.
    img_h, img_w = image.shape  # Extract height and width of the image.
    assert win_h > 0 and win_w > 0, "Window sizes must be positives."
    assert win_h < img_h and win_w < img_w, "Window sizes must be smaller than the image size."
    assert win_h % 2 == 1 and win_w % 2 == 1, "Window sizes must be odd numbers."

    # Initiate the transformed image with the correct padding.
    transformed_image = np.zeros((img_h + win_h - 1, img_w + win_w - 1), dtype=np.uint8)
    transformed_image[win_h // 2 : img_h + win_h // 2, win_w // 2 : img_w + win_w // 2] = image

    # Iterate over the image with the window size and apply the histogram 
    #  equalization for each window.
    for h in range(win_h // 2, img_h + win_h // 2):
        for w in range(win_w // 2, img_w + win_w // 2):
            # Extract the window from the image.
            window = image[
                h - win_h // 2 : h + win_h // 2 + 1,
                w - win_w // 2 : w + win_w // 2 + 1
            ]

            # Apply the histogram equalization to the window.
            transformed_image[h, w] = get_neighborhood_value(window)
            
    return transformed_image[win_h // 2 : img_h + win_h // 2, win_w // 2 : img_w + win_w // 2]
    
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
    image: np.ndarray = cv.imread(os.path.abspath(args.file_name), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not find the image file: {os.path.abspath(args.file_name)}")
    
    transformed_image = apply_local_histogram_equlization(image, args.window_size)
    show_comparison(image, transformed_image)

    image_folder = Path(args.file_name).parent
    image_name = os.path.splitext(os.path.basename(args.file_name))[0]
    cv.imwrite(str(image_folder / rf"{image_name}_transformed.jpg"), transformed_image)


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)
# ================================ End Of Main ================================ #