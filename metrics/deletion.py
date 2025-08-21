import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from metrics.utils import gkern, auc, minimal_subset,calculate_iou

class CausalMetric():

    def __init__(self, height, width, model, mode, step, device='cpu', substrate_fn=None):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.height = height
        self.width = width
        
        self.total_pixels = height*width
        
        self.model = model
        self.step = step
        self.mode = mode
        
        # set metric
        if self.mode == 'del':
            self.substrate_fn = torch.zeros_like
        elif self.mode == 'ins':
            klen = 11
            ksig = 5
            kern = gkern(klen, ksig).to(device)
            blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
            self.substrate_fn = blur
            
        # self.substrate_fn = substrate_fn # torch.zeros_like or blur

    def plotDeletionCurve(self, img_np, i, n_steps, scores, ylabel, title, save_to=None):
        # ploting function
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(img_np)
        plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
        plt.axis('off')
        

        plt.subplot(122)
        plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
        plt.title(title)
        plt.xlabel(ylabel)
        
        if save_to is not None:
            plt.savefig(save_to)
            plt.close()
        else:
            plt.show()

    def single_run(self, img_tensor, explanation, target_class_index, verbose=0, target_bbox=None, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also show the metrics
                2 - also plot the AUC Deletion curve.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """

        # set metric
        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()


        # Check if the explanation is full of zeros
        if np.all(explanation == 0):
            print("Explanation is full of zeros. Selecting pixels randomly.")
            # fill the explanation with random values, and then sort by them
            explanation = np.random.rand(*explanation.shape)
            
            # num_elements = explanation.size  # Total number of elements
            # linear_values = np.linspace(0, 1, num_elements)
            # # Reshape to the original shape of the explanation
            # explanation = linear_values.reshape(*explanation.shape)
            
            
        # set max n_steps
        n_steps = (self.total_pixels + self.step - 1) // self.step
        # initilize scores numpy
        scores = np.zeros(n_steps + 1)
        
        # ***Coordinates of pixels in order of decreasing saliency
        # Step 1: Reshape the explanation array to [num_images, total_pixels]
        reshaped_explanation = explanation.reshape(-1, self.total_pixels)
        # Step 2: Get the indices that would sort each row in ascending order
        sorted_indices = np.argsort(reshaped_explanation, axis=1)
        # Step 3: Flip the sorted indices to get them in descending order
        salient_order = np.flip(sorted_indices, axis=-1)
                
        for i in tqdm(range(n_steps+1)):
            
            results = self.model(start, verbose=False)
            step_scores = []
            
            # Check BB and its class
            if results:
                for _r,result in enumerate(results): # only has size 1 --> one image
                    for _b, box in enumerate(result.boxes):
                        score = float(box.conf.item())
                        label = int(box.cls.item())
                        
                        if label == target_class_index:

                            if target_bbox is None:
                                step_scores.append(score)
                            else:
                                bbox = box.xyxy[0].cpu().numpy()  # Access bounding box coordinates, assuming xyxy format (x1,y1 top left; x2,y2 bottom right)
                                x1, y1, x2, y2 = bbox
                                top_left_corner = (x1, y1)
                                top_right_corner = (x2, y1)
                                bottom_right_corner = (x2, y2)
                                bottom_left_corner = (x1, y2)
                                vertices = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
                                
                                iou_score = calculate_iou(polygon1=target_bbox, polygon2=vertices)
                                
                                if iou_score > 0:
                                    step_scores.append(score)            
            if step_scores:
                scores[i] = np.max(step_scores)
            else:
                scores[i] = 0 # if no prediction was made, score for that class == 0
            
            # "Delete" pixels
            if i < n_steps:
                # Get the coordinates of the pixels to be updated in this step.
                coords = salient_order[:, self.step * i:self.step * (i + 1)] # Get the coordinates of the pixels to be updated in this step.
                # Flatten the coordinates to get a 1D array.
                coords = coords.flatten()
                start.view(-1, self.total_pixels)[:, coords] = finish.view(-1, self.total_pixels)[:, coords]                
            
        # Calculate AUC
        auc_score = auc(scores)
        min_subset_pixels = minimal_subset(scores, k_threshold=0.7)
        prob_min_subset_pixels = 100*min_subset_pixels
        
        if verbose>0:
            print('AUC score:',auc_score)
            print('Minimal subset: {:.2f}%'.format(prob_min_subset_pixels))
            
            if verbose==2:
                self.plotDeletionCurve(
                    img_np= start.squeeze(0).cpu().numpy().transpose((1, 2, 0)),
                    i = i, 
                    n_steps=n_steps,
                    scores=scores,
                    ylabel=ylabel,
                    title=title
                )
        
        return scores, auc_score
    
