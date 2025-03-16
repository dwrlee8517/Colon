# trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from torch.nn import BCEWithLogitsLoss

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
    
    def loss_process(self, batch): # This is where the loss processing goes
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
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch + 1
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch+1}/{self.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        return self.model, self.optimizer, train_loss_history, val_loss_history

class ColonTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, device,
                 max_epochs, loss_fn:torch.nn.BCEWithLogitsLoss):
        """
        loss_fns: Dictionary where keys are loss names and values are tuples (loss_fn, weight).
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
