import numpy as np
import matplotlib.pyplot as plt

class PointingGame:

    def __init__(self):
        pass

    def get_activation_coverage(self, heatmap, target_bbox):
        """
        Calculate the relative sum of activations within the bounding box
        as a percentage of the sum of all activations in the heatmap.

        Parameters:
        - heatmap (2D numpy array): The explanation heatmap.
        - target_bbox (list of tuples): The target bounding box in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format.

        Returns:
        - float: The percentage of the sum of activations within the bounding box.
        """
        # Extract bounding box coordinates
        x_coords = [int(x) for x, y in target_bbox]
        y_coords = [int(y) for x, y in target_bbox]

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Extract the sub-heatmap within the bounding box
        bbox_heatmap = heatmap[ymin:ymax, xmin:xmax]

        # Calculate the sum of activations within the bounding box
        bbox_activation_sum = np.sum(bbox_heatmap)

        # Calculate the total sum of activations in the heatmap
        total_activation_sum = np.sum(heatmap)

        if total_activation_sum ==0:
            return 0
        else:
            # Calculate the relative sum of activations within the bounding box
            activation_coverage = (bbox_activation_sum / total_activation_sum) * 100
            return activation_coverage

    def forward(self, img_np, heatmap, target_bbox, predicted_bbox=None, verbose=False):
        """
        Determine if the highest value of the heatmap is within the bounding box.

        Parameters:
        - heatmap (2D numpy array): The explanation heatmap.
        - target_bbox (list of tuples): The target bounding box in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format THIS HAS TO BE THE GROUND TRUTH BB
        - predicted_bbox (list of tuples): The target bounding box in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format; used as reference to plot the difference between GT and Predicted
        - verbose (bool): If True, additional information and visualizations will be printed and plotted.

        Returns:
        - tuple: 
            - bool: True if the highest value of the heatmap is within the bounding box, False otherwise.
            - tuple: Coordinates (x, y) of the most salient pixel in the heatmap.
        """
        
        # the metric to output
        pg_output = False
        
        # Extract bounding box coordinates
        x_coords = [int(x) for x, y in target_bbox]
        y_coords = [int(y) for x, y in target_bbox]
        # print('x coords:',x_coords)
        # print('y coords:',y_coords)
        
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Find the coordinates of the highest value in the heatmap
        max_value = np.max(heatmap)
        
        # Handle case where heatmap is full of zeros
        only_zeros = True if max_value == 0 else False
                
        # max_y, max_x = np.where(heatmap == max_value)
        max_coords = np.argwhere(heatmap == max_value)

        # Check if the highest value is within the bounding box 
        if only_zeros:
            max_y, max_x = max_coords[0]          
        else:
            # (it might happen to be more than one) --TBD 
            for (max_y, max_x) in max_coords:
                if (xmin <= max_x <= xmax) and (ymin <= max_y <= ymax):
                    pg_output = True
                    break
            
    
        if verbose:
            print(f"Most salient pixel: ({max_x}, {max_y})")
            
            # Plot original image
            plt.imshow(img_np, aspect="auto")
            
            # Plot the heatmap
            plt.imshow(heatmap, cmap='jet', alpha=0.5)
            plt.colorbar()

            # Plot the bounding box
            if target_bbox is not None:
                polygon = plt.Polygon(target_bbox, closed=True, edgecolor='r', linewidth=2, fill=False)
                plt.gca().add_patch(polygon)
            if predicted_bbox is not None:
                polygon = plt.Polygon(predicted_bbox, closed=True, edgecolor='k', linewidth=1, fill=False)
                plt.gca().add_patch(polygon)
                
            # Plot the most salient pixels
            if only_zeros:
                plt.title('No object detected)')
            else:
                if pg_output:
                    _color = 'go'
                    plt.title('Heatmap with Bounding Box and Most Salient Pixel: True')
                else:
                    _color = 'ro' #'bo' for blue; 'yo' for yellow; 'co' for cyan; 'mo' for magenta; 'ko' for black; 'wo' for white
                    plt.title('Heatmap with Bounding Box and Most Salient Pixel: False')
                plt.plot(max_x, max_y, _color)  
            
            # Show the plot
            plt.show()
            
        return pg_output, (max_x, max_y)