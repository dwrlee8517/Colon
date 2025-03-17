import numpy as np
import torch
from tqdm import tqdm
import logging
import wandb
import pandas as pd

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, device, max_epochs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.current_epoch = 0
        
        # Set up a logger for the trainer
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        # Optionally add a file handler if not set up elsewhere
        if not self.logger.handlers:
            fh = logging.FileHandler("training_metrics.log")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def loss_process(self, batch):  # This is where the loss processing goes
        raise NotImplementedError
    
    def train_epoch(self):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            loss = self.loss_process(batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)
    
    def val_epoch(self):
        self.model.eval()
        losses = []
        for batch in tqdm(self.val_loader, desc="Validation"):
            loss = self.loss_process(batch)
            losses.append(loss.item())
        return np.mean(losses)
        
    def train(self):
        train_loss_history = []
        val_loss_history = []
        epoch_logs = []  # for saving to CSV later
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch + 1
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
            # Log the epoch results using logger
            log_message = f"Epoch {epoch+1}/{self.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            self.logger.info(log_message)
            
            # Log the metrics to wandb for visualization
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
            
            # Save metrics in a dict for later CSV export
            epoch_logs.append({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Optionally, save epoch_logs to a CSV file for offline analysis.
        df = pd.DataFrame(epoch_logs)
        df.to_csv("training_history.csv", index=False)
        self.logger.info("Saved training history to training_history.csv")
        
        return self.model, self.optimizer, train_loss_history, val_loss_history

class ColonTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, device,
                 max_epochs, loss_fn: torch.nn.BCEWithLogitsLoss):
        """
        loss_fn: Loss function instance.
        """
        super().__init__(model, optimizer, scheduler, train_loader, val_loader, device, max_epochs)
        self.loss_fn = loss_fn

    def loss_process(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, labels.float().unsqueeze(1))
        return loss