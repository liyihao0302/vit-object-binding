import hydra
from omegaconf import DictConfig, OmegaConf
from visualizer import Visualizer
from trainer_ce import TrainerCE
from trainer_class import TrainerClass
from trainer_instance import TrainerInstance
import wandb

@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.mode == 'train':
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerCE(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    elif cfg.mode == 'train_class':
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerClass(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    elif cfg.mode == 'train_instance':
        wandb.init(entity = cfg.wandb.entity, project = cfg.wandb.project_name, name = cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.output_dir)
        trainer = TrainerInstance(cfg, cfg.output_dir)
        trainer.train(cfg)
        wandb.finish()
    
    elif cfg.mode == 'visualize':
        visualizer = Visualizer(cfg, cfg.output_dir)
        visualizer.vis()
    elif cfg.mode == 'eval_all':
        trainer = TrainerCE(cfg, cfg.output_dir)
        trainer.eval_all(cfg)

    
if __name__ == "__main__":
    main()