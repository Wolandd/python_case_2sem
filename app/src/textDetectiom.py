import easyocr
import numpy as np
import cv2
import os
from PIL import ImageDraw, Image

def get_axis_aligned_bbox(bbox):
    """
    Convert rotated bounding box to axis-aligned bounding box.
    
    Parameters:
    - bbox: List of points defining the rotated bounding box
    
    Returns:
    - Dictionary with x_min, y_min, x_max, y_max coordinates
    """
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    return {
        'x_min': min(x_coords),
        'y_min': min(y_coords),
        'x_max': max(x_coords),
        'y_max': max(y_coords)
    }

def save_cropped_regions(image, image_path, detections):
    """Crops image regions based on detections and saves them to a new directory."""
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0]
    output_dir = os.path.join(dir_name, f"{file_name}_cropped")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (bbox, text, prob) in enumerate(detections):
        x_min, y_min, x_max, y_max = get_axis_aligned_bbox(bbox)
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        height, width = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        if x_max <= x_min or y_max <= y_min:
            continue  # Skip invalid regions
        cropped_img = image[y_min:y_max, x_min:x_max]
        output_path = os.path.join(output_dir, f"crop_{i}.png")
        cv2.imwrite(output_path, cropped_img)
    print(f"Cropped images saved to {output_dir}")

def detect_text(image):
    """
    Detect text in the image using EasyOCR.
    
    Parameters:
    - image: Input image (numpy array)
    
    Returns:
    - List of detected text with bounding boxes
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['ru', 'en'])
    
    # Ensure image is in the correct format
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Convert to RGB if needed (EasyOCR expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform text detection
    results = reader.readtext(image)
    
    # Format results
    formatted_results = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Confidence threshold
            # Convert bbox to axis-aligned coordinates
            aligned_bbox = get_axis_aligned_bbox(bbox)
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            aligned_bbox['x_min'] = max(0, min(width, aligned_bbox['x_min']))
            aligned_bbox['y_min'] = max(0, min(height, aligned_bbox['y_min']))
            aligned_bbox['x_max'] = max(0, min(width, aligned_bbox['x_max']))
            aligned_bbox['y_max'] = max(0, min(height, aligned_bbox['y_max']))
            
            formatted_results.append({
                'text': text,
                'bbox': aligned_bbox,
                'confidence': prob
            })
    
    return formatted_results

def get_bbox_coordinates(results):
    """
    Extract bounding box coordinates from detection results.
    
    Parameters:
    - results: List of detection results
    
    Returns:
    - List of bounding box coordinates
    """
    return [result['bbox'] for result in results]

if __name__ == '__main__':
    try:
        image_path = '../examples/example.png'
        results = detect_text(Image.open(image_path))
        bboxes = get_bbox_coordinates(results)
        print("Bounding box coordinates:", bboxes)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    