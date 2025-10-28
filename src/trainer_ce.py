import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import wandb # type: ignore

from tqdm import tqdm
import numpy as np


from utils.dataset import ADE20KSegmentation, FeatureExtractor, collate_fn
from utils.models import get_model
from utils.score import compute_batch_pairwise_similarity
from utils.utils import set_random_seed



class TrainerCE:
    def __init__(self, cfg, output_dir):
        
        set_random_seed(cfg.seed)
        self.model = get_model(cfg)
        
        dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="train")
        
        train_size = int(cfg.dataset.train_ratio * len(dataset))
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(dataset)))
        

        remove_indices = {7689, 8904, 8905, 20433, 20543, 21029, 22979, 23011, 23048, 24103, 24794}
        # Remove these indices from both train_indices and val_indices
        train_indices = [idx for idx in train_indices if idx not in remove_indices]
        val_indices = [idx for idx in val_indices if idx not in remove_indices]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = ADE20KSegmentation(root=cfg.dataset.data_dir, image_set="val")
        
        self.train_dataloader = DataLoader(train_dataset, batch_size = cfg.trainer.batch_size, shuffle = True, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size = cfg.trainer.batch_size, shuffle = False, num_workers = cfg.trainer.num_workers, collate_fn = collate_fn)
        
        
        print('train_dataset size:' + str(len(self.train_dataloader)))
        print('val_dataset size:' + str(len(self.val_dataloader)))
        print('test_dataset size:' + str(len(self.test_dataloader)))
        
        self.feature_extractor = FeatureExtractor(cfg, test_dataset.class_mapping)

        self.device = cfg.trainer.device
        self.model = self.model.to(self.device)
        print(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.trainer.scheduler.step_size, gamma=cfg.trainer.scheduler.gamma) 
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.output_dir = output_dir

        # Load checkpoint if exists
        if cfg.mode == 'train_ce':
            CHECKPOINT_PATH = os.path.join(output_dir, 'checkpoint.pth')
        elif cfg.mode == 'eval_all':
            CHECKPOINT_PATH = os.path.join(output_dir, 'best_checkpoint.pth')
            
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            self.best_val = checkpoint['best_val']
            self.best_test = checkpoint['best_test']
            print("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, self.start_epoch))
                
        else:
            self.start_epoch = 0
            self.best_val = 1e5
            self.best_test = 1e5
        
    
    def train(self, cfg):
        
        for epoch in range(self.start_epoch, cfg.trainer.max_epoch):
            stat_dict = {}
            
            stat_dict['train_loss'] = 0.0
            self.model.train()
            correct = 0
            total_num = 0
            total_updates = 0
            for images, masks, bboxes in tqdm(self.train_dataloader):
                features, labels, _ = self.feature_extractor.forward(images, masks, bboxes)
                labels = labels.to(self.device)
                features = features.to(self.device)
                self.optimizer.zero_grad()
                pairwise_similarity = compute_batch_pairwise_similarity(self.model, features, features)
                
                # pairwise_similarity > 0.5 -> same object -> label = 1, consider all N*N pairs
                predicted = (pairwise_similarity > 0.0).float()
                labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)#[B,N,N]
                #import pdb; pdb.set_trace()
                loss = self.criterion(pairwise_similarity.reshape(-1), labels_pairwise.reshape(-1).float())
                loss.backward()
                

                #print(f"Gradient Norm: {grad_norm.item()}")
                self.optimizer.step()
                
                correct += torch.sum(torch.triu(predicted == labels_pairwise)).item()
                stat_dict['train_loss'] += loss.item() * predicted.numel() # B * N
                total_num += torch.triu(torch.ones_like(predicted)).sum().item()
                total_updates += predicted.numel()

            stat_dict['train_acc'] = correct / total_num
            for key in ['train_loss']:
                stat_dict[key] = stat_dict[key] /total_updates
            print(f'[Epoch {epoch+1}] loss: {stat_dict["train_loss"]:.4f}, acc: {stat_dict["train_acc"]*100:.4f}%')
            # Save current checkpoint to resume training
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val': self.best_val,
                'best_test': self.best_test
            }
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)
            
            self.scheduler.step()
            
            val_performance = self.eval(cfg, epoch, split='val')
            test_performance = self.eval(cfg, epoch, split='test')
            stat_dict.update(val_performance)
            stat_dict.update(test_performance)

            if self.best_val >= stat_dict['train_loss']:
                self.best_val = stat_dict['val_loss']
                self.best_test = stat_dict['test_loss']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val': self.best_val,
                    'best_test': self.best_test
                }
                checkpoint_path = os.path.join(self.output_dir, f"best_checkpoint.pth")
                torch.save(checkpoint, checkpoint_path)
            stat_dict['epoch'] = epoch + 1
            wandb.log(stat_dict)
        
    
    def eval(self, cfg, epoch, split = 'val'):
        stat_dict = {}
        stat_dict[split + '_loss'] = 0.0

        self.model.eval()
        
        if split == 'val':
            dataloader = self.val_dataloader
        elif split == 'test':
            dataloader = self.test_dataloader
        elif split == 'train':
            dataloader = self.train_dataloader
        with torch.no_grad():
            correct = 0
            total_num = 0
            total_updates = 0
            for images, masks, bboxes in tqdm(dataloader):
                features, labels, labels_type = self.feature_extractor.forward(images, masks, bboxes)
                
                labels = labels.to(self.device)
                features = features.to(self.device)
                pairwise_similarity = compute_batch_pairwise_similarity(self.model, features, features)
                
                
                predicted = (pairwise_similarity > 0.0).float()
                labels_pairwise = labels.unsqueeze(1) == labels.unsqueeze(2)
                # test baseline
                #predicted = (labels_type.unsqueeze(1) == labels_type.unsqueeze(2)).to(predicted.device)
                #import pdb; pdb.set_trace()
                loss = self.criterion(pairwise_similarity.reshape(-1), labels_pairwise.reshape(-1).float())
                
                correct += torch.sum(torch.triu(predicted == labels_pairwise)).item()
                stat_dict[split + '_loss'] += loss.item() * predicted.numel()
                total_num += torch.triu(torch.ones_like(predicted)).sum().item()
                total_updates += predicted.numel()
                
        stat_dict[split + '_acc'] = correct / total_num
        stat_dict[split + '_loss'] = stat_dict[split + '_loss'] / total_updates
        print(f'[Eval {epoch+1}] {split} loss: {stat_dict[split + "_loss"]:.4f}, acc: {stat_dict[split + "_acc"]*100:.4f}%')
        
        return stat_dict
    
    def eval_all(self, cfg):
        stat_dict = self.eval(cfg, 0, split='test')
        
        '''
        path = os.path.join(cfg.result_dir, cfg.wandb.project_name + '_results.json')
        if os.path.exists(path):
            with open(path, "r") as f:
                my_dict = json.load(f)
        else:
            my_dict = {}
        my_dict[cfg.wandb.run_name] = stat_dict
        with open(path, "w") as f:
            json.dump(my_dict, f)
        '''
    

