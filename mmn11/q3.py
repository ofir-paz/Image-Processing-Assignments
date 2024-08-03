"""
q3.py - contains the solution for question 3 of mmn 11 - Image Processing 2024c.

The script applies error diffusion to an image, display the original 
image and the image after error diffusion and lastly saves the image after 
error diffusion.

The script receives the following arguments:
    --file_name: The image file to process. Default is "images/cameraman.jpg".
    --m, --layers: The number of layers for error diffusion. Default is 5.

The script uses the following libraries:
    - cv2
    - numpy

Please make sure to install the libraries before running the script.    

@Author: Ofir Paz
@Version: 22.07.2024
"""

# ================================== Imports ================================= #
import os
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
# ============================== End Of Imports ============================== #


# ================================= Functions ================================ #
def init_argparse() -> argparse.ArgumentParser:
    """
    Initialize the argument parser for the script.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description="Question 3 - Image Processing 2024c")
    parser.add_argument("--file_name", type=str, default=r"images\cameraman.jpg", 
                        help="The image file to process.")
    parser.add_argument("--layers", "--m", type=int, default=5, 
                        help="The number of layers for error diffusion.")
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
    cv.imshow("Before error diffusion", before_image)
    cv.imshow("After error diffusion", after_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def error_diffusion(image: np.ndarray, layers: int) -> np.ndarray:
    """
    Apply error diffusion to an image.

    Args:
        image (np.ndarray): The image to apply error diffusion to.
        layers (int): The number of layers for error diffusion.
    
    Returns:
        The image (with `dtype=np.uint8`) after error diffusion.
    """
    height, width = image.shape
    padded_image = np.zeros((height + 1, width + 1), dtype=np.uint8)
    padded_image[:-1, :-1] = image.copy().astype(np.uint8)
    
    for w in range(width):
        for h in range(height):
            val = padded_image[h, w]
            quantized_val = quantize(val, (0, 255), layers)
            error = val - quantized_val
            padded_image[h, w] = quantized_val
            padded_image[h + 1, w] += round((3 / 8) * error)
            padded_image[h, w + 1] += round((3 / 8) * error)
            padded_image[h + 1, w + 1] += round((2 / 8) * error)
    
    return padded_image[:-1, :-1]


def quantize(val: float, range: tuple[int, int], layers: int) -> int:
    """
    Quantize a value to a specific range.

    Args:
        val (float): The value to quantize.
        range (tuple[int, int]): The range to quantize to.
        layers (int): The number of layers for quantization.
    
    Returns:
        The quantized value.
    """
    min_range, max_range = range
    level_gap = (max_range - min_range) // (layers - 1)
    quantized_val = round(val / level_gap) * level_gap
    return quantized_val
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
        raise FileNotFoundError(f"Could not find the image file: {args.file_name}")
    
    error_diffused_image = error_diffusion(image, args.layers)
    show_comparison(image, error_diffused_image)

    image_folder = Path(args.file_name).parent
    image_name = os.path.splitext(os.path.basename(args.file_name))[0]
    cv.imwrite(str(image_folder / rf"{image_name}_error_diffused_m={args.layers}.png"), 
               error_diffused_image)


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)
# ================================ End Of Main ================================ #