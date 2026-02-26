from utils.utils import set_random_seed
from utils.datasets import ADE20KSegmentationActivations, collate_fn_activations
from utils.models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch
from utils.score import compute_batch_pairwise_similarity
import os
class Trainer:

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        train_dataset0 = ADE20KSegmentationActivations(cfg, mode='train0', layer=cfg.trainer.layer)
        train_dataset1 = ADE20KSegmentationActivations(cfg, mode='train1', layer=cfg.trainer.layer)
        train_dataset2 = ADE20KSegmentationActivations(cfg, mode='train2', layer=cfg.trainer.layer)
        train_dataset3 = ADE20KSegmentationActivations(cfg, mode='train3', layer=cfg.trainer.layer)
        train_dataset = train_dataset0 + train_dataset1 + train_dataset2 + train_dataset3
        
        val_dataset = ADE20KSegmentationActivations(cfg, mode='val', layer=cfg.trainer.layer)
        test_dataset = ADE20KSegmentationActivations(cfg, mode='test', layer=cfg.trainer.layer)
        
        
        self.train_dataloader = DataLoader(train_dataset, batch_size = cfg.trainer.batch_size, shuffle = True, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn_activations)
        self.val_dataloader = DataLoader(val_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn_activations)
        self.test_dataloader = DataLoader(test_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn_activations)
        self.output_dir = output_dir
        
    def init(self):
        self.device = self.cfg.device
        self.model = get_model(self.cfg).to(self.device)
        print(self.model)
        
        if os.path.exists(f'{self.output_dir}/checkpoint.pth'):
            print("Model trained, skipping training.")
            return 0
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.trainer.learning_rate, weight_decay=self.cfg.trainer.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.cfg.trainer.scheduler.step_size, gamma=self.cfg.trainer.scheduler.gamma)
        
        if self.cfg.trainer.train_mode == 'pairwise':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.cfg.trainer.train_mode == 'pointwise_class':
            self.criterion = nn.CrossEntropyLoss()

        self.start_epoch = 0
        self.best_val = 0
        self.best_test = 0

    def train(self):
        self.init()
        for epoch in range(self.start_epoch, self.cfg.trainer.max_epoch):
            stat_dict = {}
            stat_dict['train_loss'] = 0.0
            stat_dict['train_acc'] = 0.0
            self.model.train()
            acc = 0
            total_num = 0
            for activations, meta_data in tqdm(self.train_dataloader):
                
                self.optimizer.zero_grad()
                loss, acc = self.selected_loop(activations, meta_data)
                loss.backward()
                self.optimizer.step()
                
                stat_dict['train_acc'] += acc * activations.shape[0]
                stat_dict['train_loss'] += loss.item() * activations.shape[0]
                total_num += activations.shape[0]
            self.scheduler.step()
            for key in ['train_loss', 'train_acc']:
                stat_dict[key] = stat_dict[key] / total_num
            print(f'[Epoch {epoch+1}] loss: {stat_dict["train_loss"]:.4f}, acc: {stat_dict["train_acc"]*100:.4f}%')
            val_dict = self.eval('val')
            test_dict = self.eval('test')
            if val_dict['val_acc'] > self.best_val:
                self.best_val = val_dict['val_acc']
                self.best_test = test_dict['test_acc']
                print(f'New best val acc: {self.best_val:.4f}, corresponding test acc: {self.best_test:.4f}')
                self.save_model(f'{self.output_dir}/checkpoint.pth')
            

    def eval(self, split='val'):
        stat_dict = {}
        stat_dict[split + '_loss'] = 0.0
        stat_dict[split + '_acc'] = 0.0
        self.model.eval()

        if split == 'val':
            dataloader = self.val_dataloader
        elif split == 'test':
            dataloader = self.test_dataloader
        
        total_num = 0
        for activations, meta_data in tqdm(dataloader):
            with torch.no_grad():
                loss, acc = self.selected_loop(activations, meta_data)
            stat_dict[split + '_loss'] += loss.item() * activations.shape[0]
            stat_dict[split + '_acc'] += acc * activations.shape[0]
            total_num += activations.shape[0]


        for key in [split + '_loss', split + '_acc']:
            stat_dict[key] = stat_dict[key] / total_num
        print(f'[Eval] loss: {stat_dict[split + "_loss"]:.4f}, acc: {stat_dict[split + "_acc"]*100:.4f}%')
        
        return stat_dict

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        return self.model


    def selected_loop(self, activations, meta_data):
        if self.cfg.trainer.train_mode == 'pairwise':
            return self.pairwise_loop(activations, meta_data)
        elif self.cfg.trainer.train_mode in ['pointwise_class']:
            return self.pointwise_loop(activations, meta_data)
        elif self.cfg.trainer.train_mode in ['pointwise_identity']:
            return self.pointwise_identity_loop(activations, meta_data)

    def pairwise_loop(self, activations, meta_data):

        activations = torch.from_numpy(activations).to(self.device) #activations [B, n_patches, D]
        labels = torch.from_numpy(meta_data['instance_masks']).to(self.device) #[B,N]
        
        pairwise_similarity = compute_batch_pairwise_similarity(self.model, activations, activations)
        predicted = (pairwise_similarity > 0.0).float()
        #
        # baseline
        #predicted = torch.ones_like(predicted)

        
        labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)#[B,N,N]

        B, N, _ = pairwise_similarity.shape
        triu_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=self.device), diagonal=1
        ).unsqueeze(0).expand(B, -1, -1)   # <-- expand to [B,N,N]
        loss = self.criterion(pairwise_similarity.reshape(-1), labels_pairwise.reshape(-1).float())

        sim_flat   = pairwise_similarity[triu_mask]    # [K]
        pred_flat  = predicted[triu_mask]              # [K]
        label_flat = labels_pairwise[triu_mask].float()        # [K]

        #loss = self.criterion(sim_flat, label_flat)
        acc  = (pred_flat == label_flat).float().mean()
        #acc = (predicted == labels_pairwise).float().mean()

        return loss, acc

    def pointwise_loop(self, activations, meta_data):
        activations = torch.from_numpy(activations).to(self.device) #activations [B, n_patches, D]
        labels = torch.from_numpy(meta_data['seg_masks']).to(self.device) #[B,N]

        logits = self.model(activations) #[B,N,C]
        
        B, N, C = logits.shape
        logits_flat = logits.reshape(B * N, C)     # [B*N, C]
        labels_flat = labels.reshape(B * N).long()        # [B*N]
        
        loss = self.criterion(logits_flat, labels_flat)

        # acc
        preds = logits_flat.argmax(dim=-1)         # [B*N]
        acc = (preds == labels_flat).float().mean()

        preds = preds.reshape(B, N)
        predicted = preds.unsqueeze(1) == preds.unsqueeze(2) #[B,N,N]
        labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)#[B,N,N]
        acc = (predicted == labels_pairwise).float().mean()

        return loss, acc
    
    def pointwise_identity_loop(self, activations, meta_data):
        activations = torch.from_numpy(activations).to(self.device) #activations [B, n_patches, D]
        labels = torch.from_numpy(meta_data['seg_masks']).to(self.device) #[B,N]

        logits = self.model(activations) #[B,N,C]
        
        B, N, C = logits.shape
        logits_flat = logits.reshape(B * N, C)     # [B*N, C]
        labels_flat = labels.reshape(B * N).long()        # [B*N]
        
        loss = self.criterion(logits_flat, labels_flat)

        # acc
        preds = logits_flat.argmax(dim=-1)         # [B*N]
        acc = (preds == labels_flat).float().mean()

        preds = preds.reshape(B, N)
        predicted = preds.unsqueeze(1) == preds.unsqueeze(2) #[B,N,N]
        labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)#[B,N,N]
        acc = (predicted == labels_pairwise).float().mean()

        return loss, acc


        


