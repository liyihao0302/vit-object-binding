
import pandas as pd
import os
import numpy as np
import pickle as pkl
import sys
from torch.utils.data import Dataset
import cv2
sys.path.append('/workspaces/003/')
from libs.ADE20K.utils import utils_ade20k
import h5py
import matplotlib.pyplot as plt

class ADE20KSegmentation(Dataset):
    '''
    ADE20K dataset for semantic/instance segmentation
    segmentation: 150 classes + 1 background, map from original class names -> 150 classes
    instance: 1 class for each object instance

    Returns: img_id, img, seg, instance_mask
    '''
    def __init__(self, root, image_set="val", index=0):
        self.root = root
        self.image_set = image_set
        self.dataset_path = os.path.join(root, "ADE20K_2021_17_01")
        self.index_file = 'index_ade20k.pkl'
        
        df = pd.read_csv(self.dataset_path + '/objectInfo150.csv')
        self.class_dict = {}
        names = np.array(df['Name'])
        idxs = np.array(df['Idx'])
        for idx, name in enumerate(names):
            self.class_dict[name] = idxs[idx]

        # Load dataset index
        with open(os.path.join(self.dataset_path, self.index_file), 'rb') as f:
            self.index_ade20k = pkl.load(f)
        
        obj_names = self.index_ade20k['objectnames']
        

        # label mapping: full label -> 150 + 1 classes
        class_mapping = [0]
        for name in obj_names:
            name = name.replace(', ', ';')
            name = name.replace(' ', ';')
            if name in self.class_dict.keys():
                obj_id = self.class_dict[name]
            elif name == 'door' or name == 'double;door':
                obj_id = 15

            elif name == 'television;receiver;television;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box':
                obj_id = 90
            elif name == 'screen;crt;screen':
                obj_id = 142
            else:
                obj_id = 0
            class_mapping.append(obj_id)
        self.class_mapping = np.array(class_mapping) #[151]

        self.image_ids = list(range(len(self.index_ade20k['filename'])))

        if image_set == 'val':
            self.image_ids = [i for i in self.image_ids if 'validation' in self.index_ade20k['folder'][i]]
        elif image_set == 'train':
            self.image_ids = [i for i in self.image_ids if 'training' in self.index_ade20k['folder'][i]]
        
    
    def __getitem__(self, idx):
        
        image_id = self.image_ids[idx]
        full_file_name = os.path.join(self.index_ade20k['folder'][image_id], self.index_ade20k['filename'][image_id])
        info = utils_ade20k.loadAde20K(os.path.join(self.root, full_file_name))
        
        # Load Image and Segmentation Mask
        img = cv2.imread(info['img_name'])[:, :, ::-1]  # Convert BGR to RGB
        #seg = cv2.imread(info['segm_name'])[:, :, ::-1]
        instance_mask = info['instance_mask'] # [768, 1024], 0: background, 1-n: object 
        seg = info['class_mask'] # [768, 1024]
        
        seg = self.class_mapping[seg] # [768, 1024], 0: background, 1-150: object class

        
        
        return image_id, img, seg, instance_mask

    def __len__(self):
        return len(self.image_ids)

class ADE20KSegmentationActivations(Dataset):
    def __init__(self, cfg, mode='val', layer=0):
        self.cfg = cfg
        
        n_patches = cfg.dataset.num_saved_patches
        D = cfg.model.embed_dim

        mm_flat = np.memmap(self.cfg.data_extractor.activation_dir + mode + '/' +
            f"acts_layer_{layer}_{mode}.dat", mode='r', dtype=np.float32)
        N = mm_flat.size // (n_patches * D)
        
        self.memmap = mm_flat.reshape(N, n_patches, D)

        self.layer = layer
        self.size = N
        self.path = self.cfg.data_extractor.activation_dir + mode + '/' + f"labels_{mode}.h5"
        
        with h5py.File(self.path, 'r') as f:
            grp = f['labels']
            assert len(grp['img_ids']) == N, "Mismatch in number of samples between activations and labels"
        #print('N', N)

        meta_data = self.load_meta_data(0, np.arange(n_patches)) # test loading
        

    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        selected_patches = np.random.choice(self.cfg.dataset.num_saved_patches, size=self.cfg.dataset.num_samples, replace=False)
        activations = self.load_activations(idx, selected_patches) #[n_patches, D]
        meta_data = self.load_meta_data(idx, selected_patches)
        
        return activations, meta_data

    def load_activations(self, idx, selected_patches):
        
        arr = self.memmap[idx][selected_patches] #[n_patches, D]
        
        return arr
    def load_meta_data(self, idx, selected_patches):
        with h5py.File(self.path, 'r') as f:
            grp = f['labels']
            #(len(grp['img_ids']))

            # read & sub‑slice
            patch_idxs      = grp['patch_idxs'][idx][selected_patches]      # shape (num_patches,)
            img_ids         = grp['img_ids'][idx]               # shape (1,)
            seg_masks       = grp['seg_masks'][idx][selected_patches]       # shape (num_patches,)
            instance_masks  = grp['instance_masks'][idx][selected_patches]  # shape (num_patches,)
        meta_data = {
            'patch_idxs': patch_idxs,
            'img_ids': img_ids,
            'seg_masks': seg_masks,
            'instance_masks': instance_masks
        }
        return meta_data
        


def collate_fn(batch):
    img_ids = [item[0] for item in batch]
    images = [np.array(item[1]) for item in batch] #[B, H, W, 3]
    seg_masks = [np.array(item[2]) for item in batch] #[B, H, W]
    instance_masks = [np.array(item[3]) for item in batch]#[B, H, W]
    
    return img_ids, images, seg_masks, instance_masks

def collate_fn_activations(batch):
    activations = np.stack([item[0] for item in batch]) #[B, n_patches, D]
    
    meta_dicts = [item[1] for item in batch]
    # assume each dict has the same keys
    meta_data = {
        k: np.stack([d[k] for d in meta_dicts], axis=0)
        for k in meta_dicts[0].keys()
    }
        
    return activations, meta_data



