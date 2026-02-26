import hydra
from omegaconf import DictConfig, OmegaConf
from data_extractor import DataExtractor
from trainer import Trainer
from visualizer import Visualizer, prepare_one_image
import numpy as np
from utils.datasets import ADE20KSegmentation
import os
@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.mode == 'extract_and_save': # extract and save activations, labels (img_id, object_id, object_class, patch_id)
        data_extractor = DataExtractor(cfg, cfg.output_dir)
        data_extractor.extract('test')
        data_extractor.extract('val')
        data_extractor.extract('train0')
        data_extractor.extract('train1')
        data_extractor.extract('train2')
        data_extractor.extract('train3')
        
    elif cfg.mode == 'train':
        trainer = Trainer(cfg, cfg.output_dir)
        trainer.train()
    
    elif cfg.mode == 'vis':
        for img in ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png", "img6.png"]:
            image_path = f"imgs/{img}"
            visualizer = Visualizer(cfg, cfg.output_dir)
            pixel_value, patched_mask = visualizer.extract(image_path)
            issameobject_list = []
            layer_ids = [0,3,6,9,12,15,18,21,23]
            for layer in layer_ids:
                rgb, patched_mask_save, issameobject = visualizer.visualize(pixel_value, patched_mask, layer, (10,10))
                issameobject_list.append(issameobject)
            issameobject_array = np.stack(issameobject_list, axis=0)  # (num_layers, num_patches, num_patches)
            path = f"/workspaces/003/data/website/{img[:-4]}/"
            prepare_one_image(path, rgb, patched_mask_save, issameobject_array, layer_ids)
        
            
if __name__ == "__main__":
    main()