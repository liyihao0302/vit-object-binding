import itertools
import mmcv
from mmseg.datasets.pipelines import Compose

import math
import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms

def merge_scores(score_patches, img_size, crop_size, stride):
    """
    Merges overlapping attention patches into a full image attention map.

    Args:
        attention_patches (list of np.array): A list of attention maps, each of shape (crop_size, crop_size).
        img_size (int): The size of the full image (assumed to be square: img_size x img_size).
        crop_size (int): The size of each attention patch.
        stride (int): The stride for the sliding window.

    Returns:
        np.array: The merged attention map of shape (img_size, img_size).
    """
    # Initialize accumulation array and count array
    N = score_patches[0].shape[0]
    merged_score = torch.zeros((N, img_size[0], img_size[1])).to(score_patches[0].device)
    count = torch.zeros((N, img_size[0], img_size[1])).to(score_patches[0].device)
    
    if img_size[1] > img_size[0]:
        merged_score[:, :, :crop_size[1]] += score_patches[0]
        count[:, :, :crop_size[1]] += 1 if torch.count_nonzero(score_patches[0])!=0 else 0
        max_overlap = 2 * crop_size[1] - img_size[1]
        start = crop_size[1] - max_overlap
        merged_score[:, :, start:start+crop_size[1]] += score_patches[1]
        count[:, :, start:start+crop_size[1]] += 1 if torch.count_nonzero(score_patches[1])!=0 else 0
    else:
        merged_score[:, :crop_size[0], :] += score_patches[0]
        count[:, :crop_size[0], :] += 1 if torch.count_nonzero(score_patches[0])!=0 else 0
        max_overlap = 2 * crop_size[0] - img_size[0]
        start = crop_size[0] - max_overlap
        merged_score[:, start:start+crop_size[0], :] += score_patches[1]
        count[:, start:start+crop_size[0], :] += 1 if torch.count_nonzero(score_patches[1])!=0 else 0
    # Normalize by the overlap count to avoid artificial intensity increases
    merged_score /= torch.maximum(count, torch.ones_like(count))  # Avoid division by zero

    return merged_score


def inverse_transform(crop_size, stride, original_shape, scores):
    # Multiple segments in [518,518]s -> Merged map in [H,W] 
    resized_scores = []
    for score in scores:
        # Reverse the CenterPadding
        score = reverse_center_padding(score, crop_size) # [518,518] -> [512,512]
        resized_scores.append(score)
    # Merge the attentions
    if original_shape[0] < original_shape[1]:
        scaled_image_size = (crop_size[0], round(crop_size[1]/original_shape[0]*original_shape[1]))
    else:
        scaled_image_size = (round(crop_size[0]/original_shape[1]*original_shape[0]), crop_size[1])
    scores = merge_scores(resized_scores, scaled_image_size, crop_size, stride)

    transform = torchvision.transforms.Resize((original_shape[0], original_shape[1]), antialias=True) # [512,769] -> [333,500]
    scores = transform(scores).cpu().numpy()
    return scores


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        # x [1,3,512,512]
        output = F.pad(x, pads)
        # output [1,3,518,518]
        return output

def reverse_center_padding(x, original_size):
    """
    Removes the center padding applied by CenterPadding.

    Args:
        x (torch.Tensor): Padded tensor of shape (B, C, H, W).
        original_size (tuple): The original spatial size (original_H, original_W) before padding.

    Returns:
        torch.Tensor: Unpadded tensor with shape (B, C, original_H, original_W).
    """
    _, padded_h, padded_w = x.shape
    original_h, original_w = original_size

    def get_unpad(padded_size, original_size):
        pad_size = padded_size - original_size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    # Compute cropping indices
    crop_h_left, crop_h_right = get_unpad(padded_h, original_h)
    crop_w_left, crop_w_right = get_unpad(padded_w, original_w)

    # Perform cropping to remove padding
    return x[:, crop_h_left:padded_h - crop_h_right, crop_w_left:padded_w - crop_w_right]



def forward_transform(crop_size, stride, patch_size, img):
    
    if img.shape[0] < img.shape[1]:
        scaled_image_size = (crop_size[0], round(crop_size[1]/img.shape[0]*img.shape[1]))
    else:
        scaled_image_size = (round(crop_size[0]/img.shape[1]*img.shape[0]), crop_size[1])
    transform = torchvision.transforms.Resize((scaled_image_size[0], scaled_image_size[1]), antialias=True)
    
    img = np.expand_dims(img, axis=0).repeat(3, axis=0)
    img = transform(torch.tensor(img))[0]
    imgs = []
    if img.shape[0] < img.shape[1]:
        imgs.append(img[:, :crop_size[1]])
        start = -crop_size[1] + img.shape[1]
        imgs.append(img[:, start:start+crop_size[1]])
    else:
        imgs.append(img[:crop_size[0], :])
        start = -crop_size[0] + img.shape[0]
        imgs.append(img[start:start+crop_size[0], :])
    imgs_new = []
    for img in imgs:
        pad = CenterPadding(patch_size)
        img = pad(img.unsqueeze(0).unsqueeze(0))
        imgs_new.append(img)
    return imgs_new
 
def transform_masks_to_patches(masks, patch_size):
    patches = []
    for mask in masks:
        mask = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
        patches.append(mask.squeeze(0,1))
    return patches
        

def distance_point_to_bbox(x_c, y_c, bbox):
    """
    Compute the Euclidean distance from point (x_c, y_c)
    to the boundary of a bounding box (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = bbox

    # Horizontal distance
    if x_c < x_min:
        dx = x_min - x_c
    elif x_c > x_max:
        dx = x_c - x_max
    else:
        dx = 0

    # Vertical distance
    if y_c < y_min:
        dy = y_min - y_c
    elif y_c > y_max:
        dy = y_c - y_max
    else:
        dy = 0

    # Euclidean distance
    return (dx**2 + dy**2) ** 0.5