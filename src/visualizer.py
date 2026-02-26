from utils.utils import set_random_seed
from utils.datasets import ADE20KSegmentation, collate_fn
from torch.utils.data import Subset, DataLoader
from transformers import AutoModel, CLIPModel, ViTMAEForPreTraining
from tqdm import tqdm
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import h5py
import torch.nn.functional as F
import os
import time
import torchvision
from torchvision.transforms import functional as TF
import math
from data_extractor import ImageProcessor
from utils.models import get_model
import cv2
from utils.score import compute_batch_pairwise_similarity
import matplotlib.patches as patches
from skimage import measure
from PIL import Image



activations = {}
class Visualizer:
    def __init__(self, cfg, output_dir):
        
        self.cfg = cfg

        set_random_seed(cfg.seed)
        self.device = cfg.device
        device = cfg.device
        height, width = cfg.model.height, cfg.model.width
        self.processor = ImageProcessor(cfg)
        
        if self.cfg.model.name == 'openai/clip-vit-large-patch14':
            self.model = CLIPModel.from_pretrained(cfg.model.name, cache_dir=cfg.data_extractor.cache_dir).to(device)
        elif self.cfg.model.name == 'facebook/vit-mae-large':
            self.model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-large', cache_dir=cfg.data_extractor.cache_dir).to(device)
        else:
            self.model = AutoModel.from_pretrained(cfg.model.name, cache_dir=cfg.data_extractor.cache_dir).to(device)
        self.model.eval()
        if self.cfg.model.name == 'openai/clip-vit-large-patch14':
            self.register_clip_hooks()
            self.n_layers = self.model.vision_model.encoder.layers.__len__() # 24
        elif self.cfg.model.name == 'facebook/vit-mae-large':
            self.register_mae_hooks()
            self.n_layers = self.model.vit.encoder.layer.__len__()
        else:
            self.register_hooks()

            self.n_layers = self.model.encoder.layer.__len__() # 24
        
        

    def extract(self, image_path):
        img = cv2.imread(image_path)[:, :, ::-1].copy()  # Convert BGR to RGB
        padded_img, patched_mask = self.processor.pad_and_return_patched_mask(img)
        device = self.cfg.device
        pixel_values = self.processor.forward(padded_img).permute(0,3,1,2).to(device) # "pixel_values": [N,3,518,518]
        assert pixel_values.shape[0] == 1, "Batch size other than 1 not supported"
        inputs = {"pixel_values": pixel_values}
        if self.cfg.model.name == 'openai/clip-vit-large-patch14':
            inputs['input_ids'] = torch.zeros(pixel_values.shape[0], 2, dtype=torch.long).to(device)
            inputs['input_ids'][:,0] = 49406
            inputs['input_ids'][:,1] = 49407
            inputs['attention_mask'] = torch.ones(pixel_values.shape[0], 2, dtype=torch.long).to(device)
        
        with torch.no_grad():
            if self.cfg.model.name == 'facebook/vit-mae-large':
                out = self.model(**inputs, output_hidden_states=True, return_dict=True)
                mae_mask = out.mask
                patched_seg_masks = apply_mae_mask(patched_seg_masks, mae_mask)
                patched_instance_masks = apply_mae_mask(patched_instance_masks, mae_mask)
                
            else:
                _ = self.model(**inputs) #['last_hidden_state' [B,1370,1024], 'pooler_output' [B,1024]]
        self.processor.if_normalize = False
        pixel_values = self.processor.forward(padded_img).permute(0,3,1,2).to(device) # "pixel_values": [N,3,518,518]
        self.processor.if_normalize = True
        # acitvations['0']: [1, n_patches, D]
        return pixel_values[0], patched_mask
            
    def register_hooks(self):
        #import pdb; pdb.set_trace()
        for num_layer, child in self.model.encoder.layer.named_children():
            child.register_forward_hook(get_activations(num_layer))
    
    def register_clip_hooks(self):
        for num_layer, child in self.model.vision_model.encoder.layers.named_children():
            child.register_forward_hook(get_activations(num_layer))
            #num_heads = child.attn.num_heads

    def register_mae_hooks(self):
        for num_layer, child in self.model.vit.encoder.layer.named_children():
            child.register_forward_hook(get_activations(num_layer))

    def load_probe(self, layer):
        self.probe = get_model(self.cfg).to(self.device)
        print(self.probe)
        probe_path = os.path.join(
            "/workspaces/003/data/outputs_new_large/",
            f"layer_{layer}_probe_{self.cfg.probe.mode}/",
            'checkpoint.pth'
        )
        probe_state_dict = torch.load(probe_path, map_location='cpu')
        self.probe.load_state_dict(probe_state_dict)
        self.probe.eval()

    def visualize(self, pixel_value, patched_mask, layer, patch_coor=(20,30)):
        # pixel_value: [3, H, W]
        self.load_probe(layer)
        act = activations[str(layer)][:, 1:, :]  #[1, n_patches, D]

        with torch.no_grad():
            pairwise_similarity = compute_batch_pairwise_similarity(self.probe, act, act)
            issameobject = F.sigmoid(pairwise_similarity)[0]  #[n_patches, n_patches]
        patched_mask = patched_mask[0,:,:,0]
        img_mask = self.processor.expand_patched_mask(patched_mask[None, :, :])[0]

        H_image, W_image = infer_rect_hw(img_mask)
        demo_img = pixel_value[:, img_mask == 1].reshape(3, H_image, W_image)


        
        patch_idx = patch_coor_to_patch_idx(patch_coor[0], patch_coor[1], patched_mask.shape[1])

        pixel_issameobject = issameobject[patch_idx]
        H_mask, W_mask = infer_rect_hw(patched_mask)
        demo_issameobject = pixel_issameobject.reshape(patched_mask.shape)[patched_mask == 1].reshape(H_mask, W_mask)

        
        alpha = self.processor.expand_patched_mask(demo_issameobject.cpu())[0]
        rgb = demo_img.permute(1,2,0).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)

        fig, ax = plt.subplots(figsize=(10,10))

        # RGB base image, with per-pixel alpha mask
        ax.imshow(rgb, alpha=alpha)

        # Draw the red bounding box
        
        seg_mask = alpha > 0.5
        
        self.draw_boundary(ax, seg_mask, color='black', linewidth=3)
        self.draw_bounding_box(ax, patch_coor, patched_mask, color='red', linewidth=3)
        # Save
        ax.axis("off")
        plt.savefig("img_overlay.png", bbox_inches="tight", dpi=300)

        

        return np.clip(rgb, 0, 1), patched_mask.numpy(), issameobject.cpu().numpy()


        
        
        
        
    def get_bounding_box(self, patch_coor, patched_mask):
        patch_idx = patch_coor_to_patch_idx(patch_coor[0], patch_coor[1], patched_mask.shape[1])
        boundary_mask = np.zeros_like(patched_mask)
        boundary_mask[patch_coor[0], patch_coor[1]] = 1
        img_boundary_mask = self.processor.expand_patched_mask(boundary_mask[None, :, :])[0]
        ys, xs = np.where(img_boundary_mask > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        return patch_idx, y_min, y_max, x_min, x_max

    def draw_bounding_box(self, ax, patch_coor, patched_mask, color="red", linewidth=2):
        patch_idx, y_min, y_max, x_min, x_max = self.get_bounding_box(patch_coor, patched_mask)
        rect = patches.Rectangle((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1, linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    def draw_boundary(self, ax, seg_mask, color="red", linewidth=2):
        """
        Draw a continuous boundary line around regions in seg_mask.
        seg_mask: (H, W) ints
        """

        labels = np.unique(seg_mask)
        ys, xs = np.where(seg_mask != 0)

        for label in labels:
            if label == 0:
                continue  # optional: skip background

            # Binary mask for this label
            region = (seg_mask == label).astype(float)

            # Find contours at level=0.5
            contours = measure.find_contours(region, 0.5)
            
            # Draw each contour as a line
            for contour in contours:
                
                ax.plot(
                    contour[:, 1],  # x
                    contour[:, 0],  # y
                    color=color,
                    linewidth=linewidth
                )


def get_activations(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            activations[name]=(output[0].detach()) #[B, 1370, 1024]
        else:
            activations[name]=(output.detach()) #[B, 1370, 1024]
    return hook




def apply_mae_mask(patched_masks, mae_mask):

    N = patched_masks.shape[0]

    patch_flat = patched_masks.view(N, -1).to(mae_mask.device)
    bool_mask = mae_mask.bool()
    masked_vals = [
        patch_flat[b][~bool_mask[b]] for b in range(N)
    ]

    patched_masks = torch.stack(masked_vals, dim=0)

    return patched_masks

def infer_rect_hw(img_mask):
    """
    img_mask: 2D numpy array with exactly one rectangular region of ones.
    Returns (H_image, W_image, ymin, ymax, xmin, xmax)
    """
    ys, xs = np.where(img_mask == 1)

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    H_image = ymax - ymin + 1
    W_image = xmax - xmin + 1

    return H_image, W_image

def patch_coor_to_patch_idx(y, x, n_patches_width):
    """
    y: patch y coordinate
    x: patch x coordinate
    n_patches_width: number of patches along width
    """
    return y * n_patches_width + x

def save_rgb_image(img_array, path):
    """
    Save float RGB image to PNG.
    - If range is [0,1], auto scale to [0,255]
    - If range bigger than 1, just clip
    """
    arr = img_array.astype(np.float32)

    # Auto-detect normalized images
    if arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_json(obj, path):
    """Save JSON with indentation."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def prepare_one_image(
    output_dir,
    rgb_image,
    patched_mask,
    issameobject_layers,
    layer_ids,
):
    """
    output_dir: directory (data/img1/)
    rgb_image: np.array (H, W, 3) uint8
    patched_mask: (37, 37) bool or 0/1
    issameobject_layers: (num_layers, 1369, 1369) float
    layer_ids: list of layer indices [0,3,6,...]
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. save RGB image ---
    rgb_path = os.path.join(output_dir, "rgb_image.png")
    save_rgb_image(rgb_image, rgb_path)
    print(f"[Saved] {rgb_path}")

    # --- 2. save patched_mask.json ---
    patched_mask_list = patched_mask.astype(int).tolist()
    mask_path = os.path.join(output_dir, "patched_mask.json")
    save_json(patched_mask_list, mask_path)
    print(f"[Saved] {mask_path}")

    # --- 3. save issameobject.json (multi-layer) ---
    issameobject_list = []

    for li in range(len(layer_ids)):
        m = issameobject_layers[li]  # [1369, 1369]
        issameobject_list.append(m.tolist())

    issame_data = {
        "layers": layer_ids,
        "issame": issameobject_list,
    }

    issame_path = os.path.join(output_dir, "issameobject.json")
    save_json(issame_data, issame_path)
    print(f"[Saved] {issame_path}")