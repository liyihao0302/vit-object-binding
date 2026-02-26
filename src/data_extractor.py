
from utils.utils import set_random_seed
from utils.datasets import ADE20KSegmentation, collate_fn
from torch.utils.data import Subset, DataLoader
from transformers import AutoModel, CLIPModel, ViTMAEForPreTraining
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch.nn.functional as F
import os
import time
import torchvision
from torchvision.transforms import functional as TF
import math

activations = {}
class DataExtractor:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="train")
        self.class_mapping = dataset.class_mapping
        train_size = int(cfg.dataset.train_ratio * len(dataset))
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(dataset)))
        # remove indices with corrupted images.
        remove_indices = {7689, 8904, 8905, 20433, 20543, 21029, 22979, 23011, 23048, 24103, 24794, 27278}
        # Remove these indices from both train_indices and val_indices
        train_indices = [idx for idx in train_indices if idx not in remove_indices]
        val_indices = [idx for idx in val_indices if idx not in remove_indices]
        
        train_dataset = Subset(dataset, train_indices)


        N = len(train_indices)
        num_splits = 4
        chunk_size = N // num_splits

        train_subsets = []
        for i in range(num_splits):
            start = i * chunk_size
            # make sure the last split grabs any remainder
            end = (i + 1) * chunk_size if i < num_splits - 1 else N
            idxs = train_indices[start:end]
            train_subsets.append(Subset(dataset, idxs))


        val_dataset = Subset(dataset, val_indices)

        remove_indices_test = [20] # broken image in ADE20K val set
        test_dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="val")        

        test_indices = [i for i in range(len(test_dataset)) if i not in remove_indices_test]
        test_dataset = Subset(test_dataset, test_indices)
        self.batch_size = cfg.data_extractor.batch_size
        self.train_dataloader0 = DataLoader(train_subsets[0], batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        self.train_dataloader1 = DataLoader(train_subsets[1], batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        self.train_dataloader2 = DataLoader(train_subsets[2], batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        self.train_dataloader3 = DataLoader(train_subsets[3], batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size = cfg.data_extractor.batch_size, shuffle = False, collate_fn = collate_fn)
        print('train_dataset size * 4:' + str(len(self.train_dataloader0)))
        print('val_dataset size:' + str(len(self.val_dataloader)))
        print('test_dataset size:' + str(len(self.test_dataloader)))

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
        
        
        

    def extract(self, mode='train0'):
        
        self.path = self.cfg.data_extractor.activation_dir + mode + '/' + f"labels_{mode}.h5"
        if os.path.exists(self.path):
            return 0  # skip if already exists
        device = self.model.device
        if mode == 'train0':
            dataloader = self.train_dataloader0
        elif mode == 'train1':
            dataloader = self.train_dataloader1
        elif mode == 'train2':
            dataloader = self.train_dataloader2
        elif mode == 'train3':
            dataloader = self.train_dataloader3
        elif mode == 'val':
            dataloader = self.val_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        N = len(dataloader.dataset)
        n_patches = self.cfg.dataset.num_saved_patches
        D = self.cfg.model.embed_dim # 1024

        save_dir = os.path.join(self.cfg.data_extractor.activation_dir, mode)
        os.makedirs(save_dir, exist_ok=True)
        
        self.memmaps = {
            str(layer): np.memmap(
                os.path.join(save_dir, f"acts_layer_{layer}_{mode}.dat"),
                mode='w+',
                dtype=np.float32,
                shape=(N, n_patches, D)
            )
            for layer in range(self.n_layers)
        }


        for batch_idx, (img_ids, imgs, seg_masks, instance_masks) in enumerate(tqdm(dataloader)):
            
            
            for img, seg_mask, instance_mask in zip(imgs, seg_masks, instance_masks):
                pixel_values = self.processor.forward(img).permute(0,3,1,2).to(device) # "pixel_values": [N,3,518,518]
                patched_seg_masks = self.processor.forward_patch(seg_mask[:,:,None]).permute(0,3,1,2)  #[N, 1, 37, 37]
                patched_instance_masks = self.processor.forward_patch(instance_mask[:,:,None]).permute(0,3,1,2)  #[N, 1, 37, 37]
                
                
                
                
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
            
            
            Ncrops = pixel_values.shape[0] # number of crops
            
            num_patches = patched_instance_masks.view(Ncrops, -1).shape[1]  # e.g., 1369 for 37x37, 49 for MAE

            if self.cfg.dataset.num_saved_patches == Ncrops * num_patches:
                random_idxs = [np.arange(num_patches) for _ in range(Ncrops)]  # 0-1368
            else:
                replace = True if self.cfg.dataset.num_saved_patches > num_patches * Ncrops else False
                random_idxs = [np.random.choice(num_patches, size=self.cfg.dataset.num_saved_patches // Ncrops, replace=replace) for _ in range(Ncrops)]  # 0-1368
                remainder = self.cfg.dataset.num_saved_patches - (self.cfg.dataset.num_saved_patches // Ncrops) * (Ncrops - 1) # just in case not divisible
                if remainder > 0:
                    random_idxs[-1] = np.random.choice(num_patches, size=remainder, replace=True if remainder > num_patches else False)
            
            
            if self.cfg.model.name == 'facebook/vit-mae-large':
                skip_cls = 0
            else:
                skip_cls = 1
                
            random_idx_real_img = np.concatenate(
                [idxs + i * (num_patches+skip_cls) + skip_cls for i, idxs in enumerate(random_idxs)],
                axis=0
            ) # direct indices in 0- (Ncrops * (num_patches+skip_cls) -1), account for CLS token
            assert len(random_idx_real_img) == self.cfg.dataset.num_saved_patches
            random_idx_real_mask = np.concatenate(
                [idxs + i * (num_patches) for i, idxs in enumerate(random_idxs)],
                axis=0
            ) # direct indices in 0- (Ncrops * num_patches -1), without CLS
            
            self.save_activations(random_idx_real_img, batch_idx)
            self.save_labels(random_idx_real_mask, batch_idx, img_ids, patched_seg_masks, patched_instance_masks, mode)
            
                

    def save_activations(self, idxs, batch_idx):
        # 1) Stream batches of activations (dict per batch) into each memmap

        start = self.batch_size * batch_idx
        end = min(start + self.batch_size, len(self.memmaps['0']))
        N, n_patches, D = activations['0'].shape
        for i in range(len(activations)):
            
            self.memmaps[str(i)][start:end, :, :] = activations[str(i)].reshape(1, -1, D)[:, idxs, :].cpu().numpy() # [1, num_patches, 1024], skip CLS token
        
        
    def save_labels(self, idxs, batch_idx, img_ids, patched_seg_masks, patched_instance_masks, mode):
        
        img_ids = np.array(img_ids, dtype=np.int32) #[B]
        B = len(img_ids) # batch size
        num_patches = len(idxs)
        seg_masks = patched_seg_masks.reshape(1, -1)[:, idxs].to(torch.int).cpu().numpy() # [B, num_patches]
        instance_masks = patched_instance_masks.reshape(1, -1)[:, idxs].to(torch.int).cpu().numpy() # [B, num_patches]
        
        
        
        idxs = idxs.reshape(1, -1) # [B, num_patches]

        activations_start = self.batch_size * batch_idx
        
        with h5py.File(self.path, 'a') as f:
            # Create or open group
            grp = f.require_group('labels')
            assert 'img_ids' not in grp or len(grp['img_ids']) == activations_start, "Mismatch in number of samples between activations and labels"

            # Helper to create or resize+append a dataset
            def _append(name, data, dtype, chunks):
                if name not in grp:
                    # first time: create with unlimited first dim
                    shape = (0,) + data.shape[1:]
                    grp.create_dataset(
                        name, shape=shape, maxshape=(None,)+data.shape[1:], 
                        dtype=dtype, chunks=chunks, compression='lzf'
                    )
                ds = grp[name]
                old_len = ds.shape[0]
                ds.resize(old_len + B, axis=0)
                ds[old_len:old_len + B] = data

            _append('patch_idxs',     idxs,           dtype=np.int16, chunks=(B, num_patches))
            _append('img_ids',        img_ids,        dtype=np.long, chunks=(B))
            _append('seg_masks',      seg_masks,      dtype=np.int16,      chunks=(B, num_patches))
            _append('instance_masks', instance_masks, dtype=np.int16, chunks=(B, num_patches))

            
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


def get_activations(name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            activations[name]=(output[0].detach()) #[B, 1370, 1024]
        else:
            activations[name]=(output.detach()) #[B, 1370, 1024]
    return hook


class ImageProcessor:
    def __init__(self, cfg):
        self.image_mean = cfg.data_extractor.image_processor.mean
        self.image_std = cfg.data_extractor.image_processor.std
        self.shortest_edge = cfg.data_extractor.image_processor.shortest_edge
        self.crop_size = cfg.data_extractor.image_processor.crop_size
        self.patch_size = cfg.model.patch_size

        self.if_normalize = cfg.data_extractor.image_processor.normalize


    def forward(self, image):
        
        image = torch.from_numpy(image).float()/255.0
        # Input: [H,W,C] -> Output: [N, H, W, C]
        image = self.resize_image(image)
        if self.if_normalize:
            image = self.normalize_image(image)
        crops = self.crop_image(image)
        crops = torch.stack(crops) # Shape: (N, H, W, C)

        return crops

    def forward_patch(self, image):
        # Input: [H,W,C] -> Output: [N, H//patch_size, W//patch_size, C]
        # without normalization
        image = torch.from_numpy(image)
        image = self.resize_image(image, 'nearest')
        crops = self.crop_image(image)
        crops = torch.stack(crops) # Shape: (N, H, W, C)
        N, H, W, C = crops.shape
        patch_size = self.patch_size
        crops_unfolded = crops.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)  # Shape: (N, H//14, W//14, C, 14, 14)
        crops_unfolded = crops_unfolded.reshape(N, H //patch_size , W // patch_size, C, patch_size*patch_size)  # Shape: (N, H//14, W//14, C, 196)
        patch_modes, _ = torch.mode(crops_unfolded, dim=-1) # Shape: (N, H//14, W//14)
        return patch_modes

    def resize_image(self, image, mode='bicubic'):
        image = image.permute(2, 0, 1)
        h, w = image.shape[-2:]
        if h < w:
            new_h, new_w = self.shortest_edge, round(w * (self.shortest_edge / h))
        else:
            new_h, new_w = round(h * (self.shortest_edge / w)), self.shortest_edge
        if mode == 'bicubic':
            interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        elif mode == 'bilinear':
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        elif mode == 'nearest':
            interpolation = torchvision.transforms.InterpolationMode.NEAREST
        transform = torchvision.transforms.Resize((new_h, new_w), antialias=True, interpolation=interpolation)
        resized_image = transform(image)
        return resized_image.permute(1, 2, 0)

    def normalize_image(self, image):
        image = image.permute(2, 0, 1).float()
        normalized = TF.normalize(image, self.image_mean, self.image_std)
        return (normalized).permute(1, 2, 0)

    def crop_image(self, image):
        h, w = image.shape[:2]
        
        stride_h, stride_w = self.crop_size
        crop_h, crop_w = self.crop_size
        
        def compute_steps(img_dim, crop_dim, stride_dim):
            
            if img_dim <= crop_dim:
                return 1, 0
            elif img_dim > crop_dim and img_dim <= crop_dim+stride_dim:
                return 2, img_dim - crop_dim
            #   Number of steps is at least ceil((img_dim - crop_dim)/stride_dim) + 1
            steps = math.ceil(float(img_dim - crop_dim) / float(stride_dim)-1e-6) + 1
            # Recompute a uniform stride that spaces these steps evenly
            # (using integer division to keep it simple).
            if steps > 1:
                new_stride = (img_dim - crop_dim) // (steps - 1)
            else:
                new_stride = 0

            return steps, new_stride
        
        steps_h, new_stride_h = compute_steps(h, crop_h, stride_h)
        steps_w, new_stride_w = compute_steps(w, crop_w, stride_w)

        crops = []

        for i in range(steps_h):
            start_i = i * new_stride_h
            if start_i + crop_h > h:
                start_i = h - crop_h

            for j in range(steps_w):
                start_j = j * new_stride_w
                if start_j + crop_w > w:
                    start_j = w - crop_w

                # Extract the crop
                crop = image[start_i:start_i + crop_h, start_j:start_j + crop_w, :]
                crops.append(crop)

        return crops

    def pad_and_return_patched_mask(self, image):
        """
        Pad image to a square (pad with zeros), then compute a patch mask after the same
        resize + crop logic as forward_patch().
        
        Returns:
            padded_image: square numpy array (H', W', C)
            patch_mask: torch.LongTensor with 0/1 for each patch
                        (N_crops, H//patch_size, W//patch_size)
        """

        # ----- Convert to torch -----
        
        img = torch.from_numpy(image).float()  # [H,W,C]
        H, W, C = img.shape

        # ----- Step 1: pad to square -----
        side = max(H, W)
        pad_h = side - H
        pad_w = side - W

        # F.pad expects [N,C,H,W], so permute
        img_t = img.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        padded = F.pad(img_t, (0, pad_w, 0, pad_h), value=0)  # left,right,top,bottom
        padded = padded.squeeze(0).permute(1, 2, 0)  # back to [H',W',C]

        # ----- Step 2: mask image (1 for valid, 0 for padded) -----
        mask = torch.ones((H, W), dtype=torch.float32)
        mask = F.pad(mask.unsqueeze(0).unsqueeze(0),
                    (0, pad_w, 0, pad_h), value=0).squeeze()   # [H',W']
        
        patch_mask = self.forward_patch(mask.numpy()[:,:,None])  # populates self.crops
        padded_np = padded.numpy()
        
        return padded_np, patch_mask
    def expand_patched_mask(self, mask):
        """
        Expand a patch-level mask (numpy array) into per-pixel resolution by
        repeating each patch value into a patch_size x patch_size block.

        Args:
            mask (np.ndarray):
                Shape (H_p, W_p) or (N, H_p, W_p), values like 0/1.

        Returns:
            np.ndarray:
                Expanded mask with shape:
                (N, H_p * patch_size, W_p * patch_size)
        """
        import numpy as np
        patch = self.patch_size

        mask_np = np.asarray(mask)

        # Ensure shape is (N, H_p, W_p)
        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]   # add batch dim

        N, H_p, W_p = mask_np.shape
        
        # First expand patch dims: (N, H_p, W_p) → (N, H_p, W_p, patch, patch)
        expanded = mask_np[:, :, :, None, None]   # add 2 dims
        expanded = np.repeat(expanded, patch, axis=3)
        expanded = np.repeat(expanded, patch, axis=4)

        
        expanded = expanded.transpose(0, 1, 3, 2, 4)

        
        expanded = expanded.reshape(N, H_p * patch, W_p * patch)


        return expanded





def apply_mae_mask(patched_masks, mae_mask):

    N = patched_masks.shape[0]

    patch_flat = patched_masks.view(N, -1).to(mae_mask.device)
    bool_mask = mae_mask.bool()
    masked_vals = [
        patch_flat[b][~bool_mask[b]] for b in range(N)
    ]

    patched_masks = torch.stack(masked_vals, dim=0)

    return patched_masks