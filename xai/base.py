import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from skimage.segmentation import slic, mark_boundaries
import time
import os
# from PIL import Image
# import cv2

class PerturbationBase(nn.Module):
    def __init__(self, input_size, device='cpu', N=1000, p1=0.1):
        super(PerturbationBase, self).__init__()
        self.input_size = input_size
        self.device = device
        self.p1 = p1
        self.N = N
        self.rise_masks = None # created for D-CRISP

    def generate_masks_rise(self, N, s, p1, savepath='masks.npy'):
        """
        Generate random masks for the RISE algorithm.
        
        Parameters:
        - N: The number of masks to generate.
        - s: The size of the grid that the image is divided into (s x s), resolution.
        - p1: Probability of a grid cell being set to 1 (not occluded). This should be a float value in the [0, 1] range
        - savepath: The path where the generated masks are saved.
        """
        
        # set timer
        _init_time = time.time()
        
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size
        print('Cell size:',cell_size)
        
        # Generate random grid
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        print('Grid shape:', grid.shape)

        self.masks = np.empty((N, *self.input_size))
        
        # Generate masks with random shifts
        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping (with skimage)
            _upsampling = resize(grid[i], up_size, order=1, mode='reflect',anti_aliasing=False)
            self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with PIL)
            # _upsampling = Image.fromarray(grid[i]).resize(up_size.astype(int), Image.BILINEAR)
            # self.masks[i, :, :] = np.array(_upsampling)[x:x + self.input_size[0], y:y + self.input_size[1]]
            # Linear upsampling and cropping (with cv2)
            # _upsampling = cv2.resize(grid[i], (int(up_size[1]), int(up_size[0])), interpolation=cv2.INTER_LINEAR)
            # self.masks[i, :, :] = _upsampling[x:x + self.input_size[0], y:y + self.input_size[1]]
    
        # Reshape and save the masks    
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        
        # monitor time
        segs = time.time() - _init_time
        print('Total time: {:.2f}seg'.format(segs))

        
        # save
        if savepath is not None:
            np.save(savepath, self.masks)
        
        # Load masks to the specified device
        self.masks = torch.from_numpy(self.masks).float()
        self.N = N
        self.p1 = p1
        
        return segs
           
    def generate_mask_mfpp(self, img_np, N, p1, num_levels, superpixel_level_based_on_number=False, savepath='mfpp_masks.npy'):
        """
            Generate masks using the MFPP approach.

            Parameters:
            - img_np: Input image as a NumPy array.
            - N: The number of masks to generate.
            - p1: Proportion of pixels to be set to 1 (not occluded).
            - num_levels: selects the number of segmentation levels we want to use from the predefined segment_levels.
            - superpixel_level_based_on_number: determines the way in which the superpixels are selected; either 1) calculates if the sum of pixels of all superpixels exceed the number of occluded pixels given by p1 or 2) calculates the number of superpixels to be selected (even if the total area is more/less than the proportion set by p1)
            - savepath: The path where the generated masks are saved.
        """
        # Start the timer to measure the time taken for mask generation
        _init_time = time.time()
        
        # Define different levels of segmentation
        segment_levels = [50, 100, 200, 400, 800, 1600]
        # Total number of masks to be generated (a) in total (b) in each level
        num_masks_per_level = int(N)//num_levels
        num_masks = num_masks_per_level*num_levels
        # Preallocate an array to store the masks
        self.masks = np.empty((num_masks, *self.input_size), dtype='float32')

        # Initialize a counter for mask indexing
        k = 0
        # Calculate the total number of pixels in the image
        total_pixels = img_np.shape[0] * img_np.shape[1]
        # Calculate the target number of pixels to be set to 1 based on p1
        target_num_pixels = self.p1 * total_pixels

        # Iterate over the specified number of segment levels
        for level_segments in segment_levels[:num_levels]:
            # Perform SLIC segmentation on the image for the current segment level
            segments = slic(image=img_np, n_segments=level_segments)
            # Get the unique superpixel labels
            unique_segments = np.unique(segments)

            # ***(option 1) 
            if not superpixel_level_based_on_number:
                # Generate N masks for each segment level 
                for _ in tqdm(range(num_masks_per_level), desc=f'Generating filters for {level_segments} segments'):
                    
                    # Initialize a mask with zeros (all pixels occluded)
                    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype='float32')        
                    # Initialize a counter for the number of pixels set to 1
                    current_num_pixels = 0
                    # Shuffle the order of superpixel labels to ensure randomness
                    np.random.shuffle(unique_segments)

                    # Iterate over the shuffled superpixel labels
                    for superpixel in unique_segments:
                        # Break the loop if the target number of pixels is reached
                        if current_num_pixels >= target_num_pixels:
                            break
                        # Get the pixel indices for the current superpixel
                        superpixel_mask = (segments == superpixel)
                        # Set the pixels in the current superpixel to 1
                        mask[superpixel_mask] = 1
                        # Update the counter for the number of pixels set to 1
                        current_num_pixels += np.sum(superpixel_mask)
                        
                    
                    # Store the generated mask in the preallocated array
                    self.masks[k] = mask
                    k += 1
            
            # ***(option 2)
            else:
                # Calculate the number of superpixels to select based on p1
                num_superpixels = len(unique_segments)
                target_num_superpixels = int(self.p1 * num_superpixels)
                # Generate N masks for each segment level 
                for _ in tqdm(range(num_masks_per_level), desc=f'Generating filters for {level_segments} segments'):
                    # Initialize a mask with zeros (all pixels occluded)
                    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype='float32')

                    # Randomly select the target number of superpixels
                    selected_superpixels = np.random.choice(
                        unique_segments, size=target_num_superpixels, replace=False
                    )

                    # Set the pixels in the selected superpixels to 1
                    for superpixel in selected_superpixels:
                        mask[segments == superpixel] = 1

                    # Store the generated mask in the preallocated array
                    self.masks[k] = mask
                    k += 1
        
        # Reshape the masks array to include a channel dimension (num_masks, 1, height, width)
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        print('Segment-based mask shape:',self.masks.shape)
        
        # Print the total time taken for mask generation
        segs = time.time() - _init_time
        mins = int(segs // 60)
        remaining_segs = segs % 60
        print('Total time: {:.2f} seconds ({} min {:.2f} seconds)'.format(segs, mins, remaining_segs))
        
        # Save the generated masks to the specified file if a savepath is provided
        if savepath is not None:
            np.save(savepath, self.masks)
        
        # Convert the masks to a PyTorch tensor and load them onto the specified device
        self.masks = torch.from_numpy(self.masks).float()
        self.N = num_masks
        self.p1 = p1
        
        return segs
        
    def generate_combined_masks(self, img_np, N, p1, s, num_levels, rise_ratio=0.6, superpixel_level_based_on_number=False, savepath=None, rise_cache_path=None):
        """
        Generate a combination of masks using both RISE and MFPP approaches.

        Parameters:
        - img_np: Input image as a NumPy array (for MFPP).
        - N: Total number of masks to generate.
        - p1: Proportion of pixels to be set to 1 (not occluded).
        - s: The size of the grid for RISE.
        - num_levels: Number of segmentation levels for MFPP.
        - rise_ratio: Proportion of masks generated with RISE (default is 0.6).
        - savepath: The path where the generated masks are saved.
        - rise_cache_path: The path where cached RISE masks are saved/loaded.
        """
        # set timer
        
        _init_time = time.time()
        # Calculate the number of masks for each method
        rise_N = int(N * rise_ratio)
        mfpp_N = N - rise_N
        
        print(f'Generating {rise_N} RISE masks and {mfpp_N} MFPP masks.')

        # Check if RISE masks are cached
        if self.rise_masks is not None:
            print('Masks already chached!')
            segs_rise= time.time() - _init_time
            
        elif self.rise_masks is None and rise_cache_path and os.path.exists(rise_cache_path):
            print(f'Loading cached RISE masks from {rise_cache_path}')
            segs_rise= time.time() - _init_time
            self.rise_masks = torch.from_numpy(np.load(rise_cache_path)).float().to('cpu')
        else:
            # Generate RISE masks if not cached
            segs_rise = self.generate_masks_rise(N=rise_N, s=s, p1=p1, savepath=None)
            self.rise_masks = self.masks
            
            # Save the RISE masks to the cache path if provided
            if rise_cache_path:
                np.save(rise_cache_path, self.rise_masks.numpy())
                print(f'RISE masks cached at {rise_cache_path}')

        # Generate MFPP masks
        segs_mfpp = self.generate_mask_mfpp(img_np=img_np, N=mfpp_N, p1=p1, num_levels=num_levels, superpixel_level_based_on_number=superpixel_level_based_on_number, savepath=None)
        mfpp_masks = self.masks


        # Combine the two sets of masks
        combined_masks = torch.cat([self.rise_masks, mfpp_masks], dim=0)
        
        # get overall time
        segs = segs_rise + segs_mfpp
        
        # Save the combined masks if a savepath is provided
        if savepath is not None:
            np.save(savepath, combined_masks.numpy())
        
        # Set the masks attribute and other related parameters
        self.masks = combined_masks
        self.N = N
        self.p1 = p1

        print(f'Combined masks shape: {self.masks.shape}')
        
        return segs

    def generate_masks_for_level(self, img_np, N, p1, level_segments, savepath='level_masks.npy'):
        """
        Generate masks for a specific segmentation level and perform the explanation.

        Parameters:
        - img_np: Input image as a NumPy array.
        - N: The number of masks to generate.
        - p1: Proportion of pixels to be set to 1 (not occluded).
        - level_segments: Number of segments for the current level of segmentation.
        - savepath: The path where the generated masks are saved.
        """

        num_masks = N
        
        # Start the timer to measure the time taken for mask generation
        _init_time = time.time()

        # Calculate the total number of pixels in the image
        total_pixels = img_np.shape[0] * img_np.shape[1]
        # Calculate the target number of pixels to be set to 1 based on p1
        target_num_pixels = p1 * total_pixels

        # Preallocate an array to store the masks
        self.masks = np.zeros((num_masks, img_np.shape[0], img_np.shape[1]), dtype='float32')

        # Perform SLIC segmentation on the image for the current segment level
        segments = slic(image=img_np, n_segments=level_segments)
        
        # Get the unique superpixel labels
        unique_segments = np.unique(segments)

        # Generate N masks for the current segment level
        for i in tqdm(range(num_masks), desc=f'Generating masks for {level_segments} segments'):
            # Initialize a mask with zeros (all pixels occluded)
            mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype='float32')        
            # Initialize a counter for the number of pixels set to 1
            current_num_pixels = 0
            
            # Shuffle the order of superpixel labels to ensure randomness
            np.random.shuffle(unique_segments)

            # Iterate over the shuffled superpixel labels
            for superpixel in unique_segments:
                # Break the loop if the target number of pixels is reached
                if current_num_pixels >= target_num_pixels:
                    break
                # Get the pixel indices for the current superpixel
                superpixel_mask = (segments == superpixel)
                # Set the pixels in the current superpixel to 1
                mask[superpixel_mask] = 1
                # Update the counter for the number of pixels set to 1
                current_num_pixels += np.sum(superpixel_mask)

            # Store the generated mask in the array
            self.masks[i] = mask

        # Reshape the masks array to include a channel dimension (num_masks, 1, height, width)
        self.masks = self.masks.reshape(-1, 1, img_np.shape[0], img_np.shape[1])
        print(f'Generated {num_masks} masks for segmentation level {level_segments} with shape: {self.masks.shape}')

        # Print the total time taken for mask generation
        segs = time.time() - _init_time
        mins = int(segs // 60)
        remaining_segs = segs % 60
        print(f'Total time for {level_segments} segments: {segs:.2f} seconds ({mins} min {remaining_segs:.2f} seconds)')

        # Save the generated masks to the specified file if a savepath is provided
        if savepath is not None:
            np.save(savepath, self.masks)

        # Convert the masks to a PyTorch tensor and load them onto the specified device
        self.masks = torch.from_numpy(self.masks).float()
        self.N = num_masks
        self.p1 = p1
        
        return self.masks
    
    def load_masks(self, filepath_or_masks):
        """
        Load masks from a specified file path or directly from an array of masks.
        Masks are loaded in cpu (change it to consider gpu)
        
        Parameters:
        - filepath_or_masks: Either the path from where to load the masks, or the actual mask array.
        """
        
        # Check if the input is a file path or an array of masks
        if isinstance(filepath_or_masks, str):
            # If it's a string, load the masks from the file path
            self.masks = np.load(filepath_or_masks)
            self.masks = torch.from_numpy(self.masks).float().to('cpu')
            print(f'Loaded masks from {filepath_or_masks}')
        else:
            # If it's not a string, assume it's an array of masks
            self.masks = filepath_or_masks
            self.masks = self.masks.float().to('cpu')
            print(f'Loaded {len(self.masks)} masks directly')
        
        self.N = self.masks.shape[0] # Update the number of masks

