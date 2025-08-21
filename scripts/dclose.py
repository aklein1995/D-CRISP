from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import torchvision
import argparse
import torch
# import sys

from xai.drise_batch import DRISEBatch

from utils.utils import normalize_bboxes, set_seed
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#########################
# Arguments
#########################
parser = argparse.ArgumentParser()
# hw related
parser.add_argument("--device", default='cuda:2')
parser.add_argument("--model_device", default='cuda:2')
parser.add_argument("--gpu_batch", type=int, default=100)
# paths
parser.add_argument("--model_path", default='/home/tri109499/nfs_home/projects/ULTIMATE_data_training/piap_uc3/runs/detect/small/weights/best.pt')
parser.add_argument("--datadir", default='/home/tri109499/nfs_home/projects/ULTIMATE_data_training/piap_uc3/Dataset/images/val/')
parser.add_argument("--labels_dir", default='/home/tri109499/nfs_home/projects/ULTIMATE_data_training/piap_uc3/Dataset/labels/val/')
parser.add_argument("--saliency_map_dir", default='saliency_maps/')
parser.add_argument("--maskdir", default='masks/')
# data
parser.add_argument("--img_name", default='51332')
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=640)
# model
parser.add_argument("--conf_thre", type=float, default=0.7)
# seed
parser.add_argument('--seed', type=int, help='seed employed for reproducibility (random, numpy, torch, etc.)')
# mask generation
parser.add_argument("--load_masks", action='store_true')
parser.add_argument("--mask_type", choices=['close'], default='rise')
parser.add_argument("--N", type=int, default=2000)
parser.add_argument("--p1", type=float, default=0.5)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--num_levels", type=int, default=5)
parser.add_argument("--cell_size", default=50, help="Used when randomly generating the grid (regardless of image dimensions)")
parser.add_argument("--mask_ratio", type=float, default=0.5, help="Used when combining RISE and MFPP; determines RISE's masks %")
parser.add_argument("--window_size", type=int, default=64)
parser.add_argument("--target_classes", default="0")

args = parser.parse_args()
args.target_classes = [int(cls) for cls in args.target_classes.split(',')] if args.target_classes != "" else [1]

#########################
# Set seed
#########################
if args.seed is not None:
    set_seed(args.seed)
    # new directory name
    args.saliency_map_dir = os.path.join(args.saliency_map_dir,f"seed_{args.seed}") 
    # ensure target directory exist
    os.makedirs(args.saliency_map_dir, exist_ok=True)

#########################
# Define Model
#########################
model = YOLO(args.model_path, task='detect').to(args.model_device)

#########################
# Generate Explainer Instance
#########################
explainer = DRISEBatch(model=model, 
    input_size=(args.height, args.width), 
    device=args.device,
    gpu_batch=args.gpu_batch
)

#########################
# Explain & Visualize
#########################
# automatically takes all images in folder (or introduce here list manually)
imgs_name = sorted([os.path.splitext(f)[0] for f in os.listdir(args.datadir) if os.path.isfile(os.path.join(args.datadir, f))])

for img_name in imgs_name:
    img_path = args.datadir + img_name + '.jpg'
    # load image
    orig_img = Image.open(img_path).convert('RGB')
    # resize (if needed)
    resized_img = orig_img.resize((args.width, args.height), Image.LANCZOS)

    # img_np = np.array(orig_img)
    img_np = np.array(resized_img)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    tensor = preprocess(resized_img)
    tensor = tensor.unsqueeze(0).to(args.device)
    
    results = model.predict(tensor, verbose=False)
    print(f'\n\nImage: {img_name}, {len(results[0].boxes)} boxes detected')
    
    # ***generate masks
    masks = {}
    num_masks_per_level = int(args.N) // args.num_levels 
    segment_levels = [50, 100, 200, 400, 800, 1600]
    
    for l in range(args.num_levels):
        mask = explainer.generate_masks_for_level(
            img_np=img_np,
            N = num_masks_per_level,
            p1 = args.p1,
            level_segments=segment_levels[l],
        )
        masks[segment_levels[l]] = mask
    
    for _b,box in enumerate(results[0].boxes):
        # get conf
        _score = float(box.conf.item())
        
        if _score < args.conf_thre:        
            print(f'Image: {img_name} - Bounding Box {_b} conf {_score} too low; discarded!')
        else:
            # get class
            _target_class = int(box.cls.item())
            print(f'Image: {img_name} - Bounding Box {_b} class {_target_class} conf {_score}')
            # bounding box info
            bbox = box.xyxy[0].cpu().numpy()  
            x1, y1, x2, y2 = bbox
            top_left_corner = (x1, y1)
            top_right_corner = (x2, y1)
            bottom_right_corner = (x2, y2)
            bottom_left_corner = (x1, y2)
            scaled_bboxes = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
            # norm bb
            normalized_bboxes=normalize_bboxes(
                bboxes=[scaled_bboxes], #expects a list of lists/boxes
                img_height=args.height,
                img_width=args.width,
            )

            # Apply XAI
            saliency_maps = {}
            for l in segment_levels[:args.num_levels]:
                explainer.load_masks(masks[l])
                target_bbox = scaled_bboxes
                target_norm_bbox = normalized_bboxes[0] # only one image
                saliency = explainer(
                    x=tensor,
                    target_class_indices=[_target_class], # assumes a list of targets
                    target_bbox=target_bbox
                )
                saliency_maps[l] = saliency[_target_class]
            
            # Combine saliency maps
            heatmap = np.zeros_like(saliency_maps[segment_levels[0]], dtype=np.float32)
            
            for i,l in enumerate(reversed(segment_levels[:args.num_levels])):
            # for i,l in enumerate(segment_levels[:args.num_levels]):
                saliency = saliency_maps[l]
                
                # Check if the saliency map has no variation (all values are the same)
                if saliency.max() == saliency.min():
                    saliency_normalized = np.zeros_like(saliency)  # Set to zero if no variation
                else:
                    # use the density map of the mask to adjust the saliency map
                    if isinstance(masks[l], torch.Tensor):
                        density_map = np.sum(masks[l].squeeze().cpu().numpy() != 0, axis=(0))  # Convert to numpy array first
                        # print(f'maks shape {masks[l].shape} and density shape {density_map.shape} saliency of shape: {saliency.shape}')
                    else:
                        density_map = np.sum(masks[l] != 0, axis=(0))  
                    
                    # Avoid division by zero: replace zeros in density_map with a small constant (1e-6) to prevent NaN values or extreme results
                    density_map[density_map == 0] = 1
                    saliency /= density_map
                    
                    # Normalize the saliency map for the current level
                    saliency_normalized = (saliency - saliency.min()) / (saliency.max() - saliency.min())

                # Add the (normalized) saliency map to the heatmap
                heatmap += saliency_normalized
                
                # If it's not the first level, multiply the heatmap with the current saliency map
                if i != 0:
                    heatmap *= saliency_normalized
            
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            else:
                heatmap = np.zeros_like(heatmap)  # Set to zeros if max is zero to avoid NaN


            # ***save saliency map
            filename = f'img{img_name}' + f'_class_{_target_class}_bb_{target_norm_bbox}.npy'
            saliency_map_path = os.path.join(args.saliency_map_dir, filename)
            # save it
            np.save(saliency_map_path, heatmap)
