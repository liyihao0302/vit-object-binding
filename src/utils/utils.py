from torchvision.datasets import VOCSegmentation
import os
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import io
import matplotlib.colors as mcolors




def set_random_seed(seed_value):

    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_attention(img, attention, layer, vis_cfg, transformed_mask):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, alpha=0.5)  # Overlay original image
    #how to set cmap to be in the range of [0,1]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    ax.imshow(attention, cmap="jet", alpha=0.5, norm=norm)  # Overlay attention heatmap
    if vis_cfg.mode == 'patch' or vis_cfg.mode == 'object':
        where = np.where(transformed_mask[0]>0)
        y0 = np.min(where[0])
        y1 = np.max(where[0])
        x0 = np.min(where[1])
        x1 = np.max(where[1])
        ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], color="red", linewidth=3)
    ax.axis("off")
    plt.title(f"Layer {layer}")
    return fig

def plot_attentions(image, attentions, layers, vis_cfg, transformed_mask, file_name, gif=True, duration=500, loop=0):
    if gif == False:
        fig = plot_attention(image, attentions[0], layers[0], vis_cfg, transformed_mask)
        plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    elif gif == True:
        frames = []  # Store frames in memory
        count = 0
        for attention, layer in zip(attentions, layers):
            fig = plot_attention(image, attention, layer, vis_cfg, transformed_mask)
            # Save figure to a BytesIO buffer instead of a file
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            #plt.savefig(file_name+f'_{count}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Convert buffer image to PIL format and append to frames
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame)
            
            
            count+=1
        # Save all frames as a GIF
        frames[0].save(file_name+'.gif', save_all=True, append_images=frames[1:], duration=duration, loop=loop)


def plot_segmentation(image, mask, segmented_image, file_name):

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    # Display the segmentation mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="viridis")
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image, cmap="gray")
    plt.title("Segmentation Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(file_name)