import os
import time
import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.train.trainer import ColonTrainer
from src.utils.dataset import RealColonDataset
from src.utils.helpers import choose_gpu_with_cuda_visible_devices, plot_and_save_training_validation_loss, set_random_seed

import wandb
import logging
import matplotlib.pyplot as plt

# Set up Python logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO or WARNING for less verbosity in production
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="/radraid2/dongwoolee/Colon/configs", config_name="config1")
def main(cfg: DictConfig):

    output_dir = os.getcwd() # Hydra's output directory
    
    # Print and log the configuration
    config_str = OmegaConf.to_yaml(cfg)
    print(config_str)
    logger.info("Configuration:\n%s", config_str)

    # Initialize wandb with the full configuration
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))
    logger.info("Wandb run initialized.")

    device = torch.device('cuda')
    set_random_seed(cfg.seed)

    # Instantiate model, optimizer, and scheduler
    model:Module = hydra.utils.instantiate(cfg.model)
    model.to(device)
    model_name = type(model).__name__
    logger.info("Loaded Model: %s", model_name)

    optimizer:Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler:LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    loss_fn:Module = hydra.utils.instantiate(cfg.loss)

    train_dataset:Dataset = hydra.utils.instantiate(cfg.dataset.train)
    valid_dataset:Dataset = hydra.utils.instantiate(cfg.dataset.valid)
    test_dataset:Dataset  = hydra.utils.instantiate(cfg.dataset.test)
    logger.info("Train Dataset: %s, Valid Dataset: %s", len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Print shapes of first batch for debugging
    for images, labels in train_dataloader:
        logger.debug("Train batch - images: %s, labels: %s", images.shape, labels.shape)
        break

    for images, labels in valid_dataloader:
        logger.debug("Validation batch - images: %s, labels: %s", images.shape, labels.shape)
        break

    trainer = ColonTrainer(model, optimizer, scheduler, train_dataloader, valid_dataloader, device, cfg.epochs, loss_fn)
    model, optimizer, train_loss_history, test_loss_history = trainer.train()

    # Save results in Hydra's output directory
    run_id = time.strftime("%Y%m%d-%H%M%S")
    save_filepath = os.path.join(output_dir, f"{model_name}_{run_id}.pth")
    save_plotpath = os.path.join(output_dir, f"{model_name}_{run_id}.png")
    config_save_path = os.path.join(output_dir, f"{model_name}_{run_id}_config.yaml")
    
    # Save and log the loss plot
    logger.info("Saving loss plot...")
    plot_and_save_training_validation_loss(train_loss_history, test_loss_history, save_plotpath)
    logger.info("Loss plot saved to %s", save_plotpath)

    # Log loss plot to wandb as an image artifact
    wandb.log({"loss_plot": wandb.Image(save_plotpath)})

    # Save model checkpoint locally
    torch.save(model.state_dict(), save_filepath)
    logger.info("Model checkpoint saved to %s", save_filepath)

    # Log model checkpoint as a wandb artifact
    model_artifact = wandb.Artifact(model_name, type="model")
    model_artifact.add_file(save_filepath)
    wandb.log_artifact(model_artifact)

    # Save the current configuration alongside the checkpoint
    OmegaConf.save(config=cfg, f=config_save_path)
    logger.info("Configuration saved to %s", config_save_path)
    config_artifact = wandb.Artifact(f"{model_name}_config", type="config")
    config_artifact.add_file(config_save_path)
    wandb.log_artifact(config_artifact)

    wandb.finish()
    logger.info("Training finished and wandb run closed.")
if __name__=="__main__":
    main()