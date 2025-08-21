import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import re
import ast
import random
import torch

def set_seed(seed: int):
    """Set seed for random, numpy & torch"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scale_bbox(normalized_bbox, img_height, img_width):
    """
    Convert a single normalized bounding box to absolute pixel coordinates.

    Parameters:
    - normalized_bbox: List of tuples representing the normalized bounding box coordinates.
    - img_height: Height of the image.
    - img_width: Width of the image.

    Returns:
    - List of tuples containing scaled bounding box coordinates.
    """
    scaled_bbox = [(x * img_width, y * img_height) for x, y in normalized_bbox]
    return scaled_bbox


def extract_bbox_class_and_path(saliency_map_dir,filename):
    """
    Extracts the bounding box coordinates, class type, and constructs the saliency map path.

    Parameters:
    - saliency_map_dir: Directory where the saliency map files are stored.
    - filename: The filename containing the bounding box coordinates and class type.

    Returns:
    - saliency_map_path: The complete path to the saliency map.
    - target_bbox: The list of tuples representing the bounding box coordinates.
    - class_type: The class type extracted from the filename.
    """
    # Extract the bounding box string from the filename using regex
    bbox_str = re.search(r'\[\(.*?\)\]', filename).group(0)
    
    # Convert the string representation of the list of tuples to an actual list of tuples
    target_bbox = ast.literal_eval(bbox_str)
    
    # Extract the class type using regex
    class_type = re.search(r'class_(\d+)_bb', filename).group(1)
    
    # Construct the full path
    if 'npy' in filename:
        saliency_map_path = saliency_map_dir + '/' + filename
    else:
        saliency_map_path = saliency_map_dir + '/' + filename + '.npy'
        
    return target_bbox, class_type, saliency_map_path

def normalize_bboxes(bboxes, img_height, img_width):
    """
    Normalize bounding box coordinates.

    Parameters:
    - bboxes: List of bounding boxes, each represented by a list of vertices in absolute coordinates.
    - img_height: Height of the image.
    - img_width: Width of the image.

    Returns:
    - List of normalized bounding boxes.
    """
    normalized_bboxes = []
    for bbox in bboxes:
        
        # Extract coordinates from the list of vertices
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        
        # Normalize coordinates
        normalized_bbox = [(x1 / img_width, y1 / img_height), 
                           (x2 / img_width, y1 / img_height), 
                           (x2 / img_width, y2 / img_height), 
                           (x1 / img_width, y2 / img_height)]
        
        normalized_bboxes.append(normalized_bbox)
    return normalized_bboxes

def load_and_convert_bboxes(txt_file_path, img_height, img_width, target_class=None, return_class=False, isCOCO=False):
    """
    Load bounding boxes from a text file, convert the coordinates to absolute values based on processed image size,
    and return both the normalized and scaled bounding box coordinates.

    Parameters:
    - txt_file_path: Path to the text file containing bounding box coordinates.
    - img_height: Original height of the image.
    - img_width: Original width of the image.
    - target_class: Class ID of the target object (if None, does not filter by class).
    - return_class: Whether to return class IDs alongside bounding boxes.
    - isCOCO: used to do some mappings with the COCO dataset

    Returns:
    - List of tuples containing scaled (and normalized) bounding box coordinates for the processed image size; the id of the classes can also be provided
    """
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    
    if isCOCO:
        import json
        with open('utils/coco_mapping_91to80.json', 'r') as f:
            class_mapping = json.load(f)
    

    bboxes = []
    normalized_bboxes = []
    classes = []
    
    for line in lines:
        parts = list(map(float, line.split()))
        class_id = int(parts[0])
        
        if isCOCO:
            class_id = class_mapping[str(class_id)] - 1        

        # print('Class id in .txt:',class_id)
        # print('Target class from yolo:',target_class)
        
        if target_class is None or class_id == target_class:
            classes.append(class_id)
                
            # Extract normalized vertices
            normalized_vertices = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
            
            # Scale vertices to image size
            scaled_vertices = [
                (
                    parts[i] * img_width,  # Scale x
                    parts[i + 1] * img_height   # Scale y
                )
                for i in range(1, len(parts), 2)
            ]
                        
            # append data
            normalized_bboxes.append(normalized_vertices)
            bboxes.append(scaled_vertices)
            
    if return_class:
        return bboxes, normalized_bboxes, classes
    else: 
        return bboxes, normalized_bboxes