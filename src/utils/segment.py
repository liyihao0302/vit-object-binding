import numpy as np
import urllib
from functools import partial
from mmseg.apis import init_segmentor, inference_segmentor
from utils.transforms import CenterPadding
from PIL import Image
def create_segmenter(cfg, backbone_model): # add segment_head
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model


def render_segmentation(segmentation_logits, dataset, colormap):
    
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits]
    return Image.fromarray(segmentation_values)

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
