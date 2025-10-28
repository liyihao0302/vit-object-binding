import os
import numpy as np

import copy

import torch
import torch.nn.functional as F
import torch.nn as nn


import dinov2.eval.segmentation.models
import math
import mmcv
from mmcv.runner import load_checkpoint

from PIL import Image
import matplotlib.pyplot as plt

import dinov2.eval.segmentation.utils.colormaps as colormaps

from utils.transforms import inverse_transform, forward_transform, transform_masks_to_patches
from utils.segment import create_segmenter, render_segmentation, inference_segmentor, load_config_from_url
from utils.utils import set_random_seed, plot_attentions, plot_segmentation
from utils.score import compute_attention, compute_issameobject
from utils.dataset import ADE20KSegmentation
from utils.models import get_model
import os
os.environ["TORCH_HOME"] = "/workspaces/003/data/dinov2"

activations = {}
class Visualizer():
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[cfg.model.backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"

        self.backbone_model = torch.hub.load(repo_or_dir=cfg.model.model_dir, model=backbone_name, source = 'local')
        self.backbone_model.eval()
        self.backbone_model.to(self.cfg.device)
        

        HEAD_SCALE_COUNT = cfg.model.head_scale# more scales: slower but better results, in (1,2,3,4,5)
        HEAD_DATASET = cfg.dataset.name # in ("ade20k", "voc2012")
        HEAD_TYPE = cfg.model.head_type # in ("ms, "linear")

        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

        cfg_str = load_config_from_url(head_config_url)
        mmcfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        #import pdb; pdb.set_trace()
        self.mmcfg = mmcfg
        if HEAD_TYPE == "ms":
            mmcfg.data.test.pipeline[1]["img_ratios"] = mmcfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
            print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

        self.model = create_segmenter(mmcfg, backbone_model=self.backbone_model)
        load_checkpoint(self.model, head_checkpoint_url, map_location=self.cfg.device)
        self.model.to(self.cfg.device)
        self.model.eval()
        
        # Load the dataset
        data_dir = self.cfg.dataset.data_dir
        self.dataset = ADE20KSegmentation(root=data_dir, image_set="val", index=self.cfg.dataset.index)

        if self.cfg.mode == 'eval':
            self.probes = []
            for num_layer in self.cfg.model.num_layer:
                checkpoint_dir =  f'/workspaces/003/src/../data/outputs2/{cfg.probe.mode}_/' + f'layer_{num_layer}_probe_{cfg.probe.mode}/'
                CHECKPOINT_PATH = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                probe = get_model(self.cfg)
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.cfg.device, weights_only=True)
                probe.load_state_dict(checkpoint['model_state_dict'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_val = checkpoint['best_val']
                self.best_test = checkpoint['best_test']
                probe.eval()
                probe.to(self.cfg.device)
                print("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, self.start_epoch))
                self.probes.append(copy.deepcopy(probe))

    def vis(self):
        
        # Add hooks to layer
        for num_layer, child in self.backbone_model.blocks.named_children():
            if int(num_layer) in self.cfg.model.num_layer:
                child.register_forward_hook(get_activations(num_layer))
                num_heads = child.attn.num_heads
        self.num_heads = num_heads
        patch_size = self.backbone_model.patch_size
        DATASET_COLORMAPS = {
            "ade20k": colormaps.ADE20K_COLORMAP,
            "voc2012": colormaps.VOC2012_COLORMAP,
        }
        image, mask, bbox = self.dataset[self.cfg.dataset.index]
        
        image = Image.open('/workspaces/003/src/results/example.png').convert('RGB')
        array = np.array(image)[:, :, ::-1] # BGR
        #import pdb; pdb.set_trace()
        #mask = np.array(mask) #[366,500]
        #mask[mask == 255] = 0 # Convert ignore regions to background
        
        segmentation_logits = inference_segmentor(self.model, array)[0]
        colormap = DATASET_COLORMAPS[self.cfg.dataset.name]
        segmented_image = render_segmentation(segmentation_logits, self.cfg.dataset.name, colormap)

        #plot_segmentation(image, mask, segmented_image, self.cfg.result_dir+'image_voc.png')
        

        
        attentions = []
        for l_i, num_layer in enumerate(self.cfg.model.num_layer):
            # Visualize object-level attention map
            activation = activations[str(num_layer)] #[1, 1370, 1024]
            attns = []
            with torch.no_grad():
                for idx, act in enumerate(activation):
                    # we keep only the output patch attention

                    if self.cfg.vis.mode == 'patch':
                        patch_x, patch_y = self.cfg.vis.patch_x, self.cfg.vis.patch_y
                        n = patch_y * int(math.sqrt(act.shape[1]-1)) + patch_x + 1
                        if idx == self.cfg.vis.window_idx:
                            # act[:, 1:] [1,1369, 1024]
                            attn = compute_issameobject(self.probes[l_i], act[:,1:], n)
                        else:
                            attn = torch.zeros((1, act.shape[1]-1)).to(act.device)
                    elif self.cfg.vis.mode == 'object':
                        label = bbox[self.cfg.vis.object_idx]['class']
                        xmin, ymin, xmax, ymax = bbox[self.cfg.vis.object_idx]['bbox']
                        
                        # {'class': 'person', 'bbox': (xmin=74, ymin=1, xmax=272, ymax=462)}
                        bbox_mask = np.zeros_like(segmentation_logits)
                        bbox_mask[ymin-1:ymax, xmin-1:xmax] = 1
                        object_mask = np.where(segmentation_logits==label, 1, 0)
                        object_mask = object_mask * bbox_mask
                        masks = forward_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, patch_size, object_mask)
                        patches = transform_masks_to_patches(masks, patch_size)
                        
                        attn = torch.mean(compute_belonging(self.probe, act)[0, :, np.where(patches[idx].reshape(-1,1)>0.5)[0], 1:], dim=1)
                        
                    patch_length = int(math.sqrt(attn.shape[1]))
                    attn = attn.reshape(attn.shape[0], patch_length, patch_length)
                    attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0]
                    attns.append(attn)

            attention = inverse_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, array.shape, attns)
            if self.cfg.vis.head == -1:
                attention = np.mean(attention, axis=0)
            else:
                attention = attention[self.cfg.vis.head]
            attentions.append(attention)
            if self.cfg.vis.gif==False:
                break

            if self.cfg.vis.mode == 'patch':
                masks = []
                for i in range(len(activation)):
                    mask = torch.zeros((1, patch_length * patch_size, patch_length * patch_size))
                    patch_x, patch_y = self.cfg.vis.patch_x, self.cfg.vis.patch_y
                    if i == self.cfg.vis.window_idx:
                        mask[:, patch_y * patch_size: (patch_y + 1) * patch_size, patch_x * patch_size: (patch_x + 1) * patch_size] = 1
                    masks.append(mask)
                transformed_mask = inverse_transform(self.mmcfg.model.test_cfg.crop_size, self.mmcfg.model.test_cfg.stride, array.shape, masks)

            elif self.cfg.vis.mode == 'object':
                transformed_mask = np.expand_dims(object_mask, axis=0)
            else:
                None
        
        i = self.cfg.vis.head if self.cfg.vis.head != -1 else 'avg'
        plot_attentions(image, attentions, self.cfg.model.num_layer, self.cfg.vis, transformed_mask, self.cfg.result_dir+self.cfg.vis.mode+f'/attention_map_{i}', gif=self.cfg.vis.gif)
    

#Extracting activations from the backbone
def get_activations(name):
    def hook(model, input, output):
        if activations.get(name) is None:
            activations[name] = []
        activations[name].append(output.detach())
    return hook