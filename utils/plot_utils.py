import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def plot_saliency_map(img_name, saliency_map_path):
    """
    Plots the saliency map for a given image name.

    Parameters:
    - img_name: Name of the image to plot the saliency map for.
    - saliency_map_path: Path to the saliency map npy file.
    """
    saliency = np.load(saliency_map_path)
    # saliency = saliency[795]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Saliency Map for {img_name}')
    plt.axis('off')
    plt.show()


def plot_image_with_bboxes(img_np, bboxes, title=None, save_to=None):
    """
    Plots an image with bounding boxes.

    Parameters:
    - img_np: NumPy array representing the image.
    - bboxes: List of bounding boxes, each represented by a list of vertices in pixel coordinates.
    """
    if isinstance(img_np, torch.Tensor):
        # Move to CPU, convert to NumPy and transpose from (C, H, W) to (H, W, C)
        img_np = img_np.cpu().numpy().transpose((1, 2, 0))

    # Display the image
    plt.imshow(img_np)
    
    # Iterate over each bounding box
    for bbox in bboxes:
        # Create a polygon from the bounding box vertices
        polygon = plt.Polygon(bbox, closed=True, edgecolor='r', linewidth=2,fill=None)
        # Add the polygon to the plot
        plt.gca().add_patch(polygon)
    
    # Set the title of the plot
    if title is not None:
        plt.title(title)
    else:
        plt.title("Image with Bounding Boxes")
    # Remove the axes for a cleaner look
    plt.axis('off')
    if save_to is not None:
        plt.savefig(fname=save_to)
    # Display the plot
    plt.show()

def plot_saliency_on_image(height, width, saliency_map, img, img_name='img_name', target_class_id=0):
    """
    Plots the saliency map overlaid on the original image.

    Parameters:
    - height, width: Dimensions to which the original image is resized.
    - img_path: Path to the original image.
    - img_name: Name of the image to plot the saliency map for.
    - saliency_map_path: Path to the saliency map npy file.
    - target_class_id: ID of the class we are interested to plot the saliency maps for.
    """
    # Load the saliency map
    if not isinstance(saliency_map, np.ndarray):
        saliency = np.load(saliency_map)
    else:
        saliency = saliency_map
    
    if not isinstance(img, np.ndarray):
        original_img = Image.open(img)
        # Resize it according to the dimensions in which the heatmaps were created
        resized_img = original_img.resize((width, height), Image.LANCZOS)
        img_np = np.array(resized_img)
    else:
        img_np = img

    # *** Plotting
    plt.figure(figsize=(5, 5))
    plt.title(f'Saliency Map for {img_name} and target class id:{target_class_id}')
    # 1. Display the original image
    plt.imshow(img_np)
    plt.axis('off')
    # 2.Overlay the saliency map
    plt.imshow(saliency, cmap='jet', alpha=0.5)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()

def plot_saliency_and_targetbb_on_image(height, width, saliency_map, img, img_name='img_name', target_class_id=0, target_bbox=None, figsize=None, display_title=True, save_to=None):
    """
    Plots the saliency map overlaid on the original image along with the target bounding box.

    Parameters:
    - height, width: Dimensions to which the original image is resized.
    - img: Numpy or Path to the original image.
    - img_name: Name of the image to plot the saliency map for.
    - saliency_map: Numpy or Path to the saliency map npy file.
    - target_class_id: ID of the class we are interested to plot the saliency maps for.
    - target_bbox: List of tuples representing the normalized vertices of the bounding box.
    """

    # Load the saliency map
    if not isinstance(saliency_map, np.ndarray):
        saliency = np.load(saliency_map)
    else:
        saliency = saliency_map
    
    # Load the original image
    if not isinstance(img, np.ndarray):
        original_img = Image.open(img)
        # Resize it according to the dimensions in which the heatmaps were created
        resized_img = original_img.resize((width, height), Image.LANCZOS)
        img_np = np.array(resized_img)
    else:
        img_np = img

    # Plotting
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else (5, 5))
    
    if display_title:
        plt.title(f'Saliency Map for {img_name} and target class id:{target_class_id}')
    # 1. Display the original image
    # plt.imshow(img_np)
    # plt.axis('off')
    img_plot = ax.imshow(img_np)
    ax.axis('off')
    
    # 2. Overlay the saliency map
    # plt.imshow(saliency, cmap='jet', alpha=0.5)
    saliency_plot = ax.imshow(saliency, cmap='jet', alpha=0.5)
    
    # 3. Plot the bounding box if provided
    if target_bbox is not None:
        polygon = plt.Polygon(target_bbox, closed=True, edgecolor='r', linewidth=2, fill=False)
        # plt.gca().add_patch(polygon)
        ax.add_patch(polygon)
        
    plt.colorbar(saliency_plot, ax=ax, fraction=0.046, pad=0.04, shrink=0.65)
    
    if save_to is not None:
        plt.savefig(fname=save_to)
    plt.tight_layout()
    plt.show()
