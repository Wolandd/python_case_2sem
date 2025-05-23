import cv2
import numpy as np
from PIL import Image
from .imgRotat import check_rotation, deskew

def apply_binarization(image, threshold_value=128):
    """
    Apply binary thresholding to the image.

    Parameters:
    - image: Input image (grayscale).
    - threshold_value: Threshold value for binarization.

    Returns:
    - Binary image.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image


def apply_canny_edge(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to the image.

    Parameters:
    - image: Input image (grayscale).
    - low_threshold: Lower threshold for the hysteresis procedure.
    - high_threshold: Upper threshold for the hysteresis procedure.

    Returns:
    - Edge-detected image.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def apply_inverse(image):
    """
    Apply inverse transformation to the image.

    Parameters:
    - image: Input image (grayscale).

    Returns:
    - Inverted image.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply inverse transformation
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def apply_sharpen(image):
    """
    Apply sharpening to the image.

    Parameters:
    - image: Input image (grayscale).

    Returns:
    - Sharpened image.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 10, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def apply_morphological(image):
    """
    Apply morphological operations to the image.

    Parameters:
    - image: Input image (grayscale).

    Returns:
    - Morphological image.
    """
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations
    kernel = np.ones((1, 1), np.uint8)
    morphological_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return morphological_image


def preprocess_image(image):
    """
    Preprocess image for better text detection by combining multiple image processing techniques.
    
    Parameters:
    - image: Input image (PIL Image, numpy array, or path to image)
    
    Returns:
    - Preprocessed image as numpy array
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):  # PIL Image
        image = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing steps
    # 1. Sharpen the image
    image = apply_sharpen(image)
    
    # 2. Apply morphological operations
    image = apply_morphological(image)
    
    # 3. Apply binarization
    image = apply_binarization(image)
    
    # 4. Apply inverse if needed (for dark text on light background)
    image = apply_inverse(image)
    
    return image


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.rotation_angle = 0
        self.deskew_angle = 0

    def set_image(self, image):
        """Set the input image and convert to numpy array if needed"""
        if isinstance(image, str):
            self.original_image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            self.original_image = np.array(image)
            if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        else:
            self.original_image = image.copy()
        
        # Initialize display image as a copy of original
        self.display_image = self.original_image.copy()
        return self

    def preprocess(self):
        """Apply all preprocessing steps to the image"""
        if self.original_image is None:
            raise ValueError("No image set. Call set_image() first.")

        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.original_image.copy()

        # Apply preprocessing steps
        processed = apply_sharpen(gray_image)
        processed = apply_morphological(processed)
        processed = apply_binarization(processed)
        processed = apply_inverse(processed)

        self.processed_image = processed
        return self

    def rotate_for_display(self):
        """Rotate image for display purposes only"""
        if self.display_image is None:
            raise ValueError("No image set. Call set_image() first.")

        try:
            # Apply rotation for display
            rotated = check_rotation(self.display_image)
            if rotated is not None:
                self.display_image = rotated
                
            # Apply deskew
            deskewed = deskew(self.display_image)
            if deskewed is not None:
                self.display_image = deskewed
                
            return self
        except Exception as e:
            print(f"Error during rotation/deskew: {str(e)}")
            return self

    def get_processed_image(self):
        """Get the processed image for model input"""
        if self.processed_image is None:
            raise ValueError("No processed image available. Call preprocess() first.")
        return self.processed_image

    def get_display_image(self):
        """Get the image for display purposes"""
        if self.display_image is None:
            raise ValueError("No image set. Call set_image() first.")
        # Convert BGR to RGB for display
        if len(self.display_image.shape) == 3 and self.display_image.shape[2] == 3:
            return cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        return self.display_image

    def get_original_image(self):
        """Get the original input image"""
        if self.original_image is None:
            raise ValueError("No image set. Call set_image() first.")
        return self.original_image


if __name__ == "__main__":
    image = cv2.imread('./examples/example.png', cv2.IMREAD_GRAYSCALE)
    binary_image = apply_binarization(image)
    canny_edge_image = apply_canny_edge(image)
    inversed_image = apply_inverse(image)
    sharp_image = apply_sharpen(image)
    morphological_image = apply_morphological(image)

    # Combine methods to enhance text on the image
    combined_image = apply_inverse(image)
    combined_image = apply_sharpen(combined_image)
    combined_image = apply_morphological(combined_image)
    combined_image = apply_binarization(combined_image)
    cv2.imwrite('./examples/combined_image.png', combined_image)

    # Save or display the processed images
    cv2.imwrite('./examples/binary_image.png', binary_image)
    cv2.imwrite('./examples/canny_edge_image.png', canny_edge_image)
    cv2.imwrite('./examples/inversed_image.png', inversed_image)
    cv2.imwrite('./examples/sharp_image.png', sharp_image)
    cv2.imwrite('./examples/morphological_image.png', morphological_image)

