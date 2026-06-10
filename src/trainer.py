import torch


class Trainer:
    def __init__(self, args, model, optimizer, criterion, device):
        self.args = args
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        pass

    def _train_step(self, x, y):
        pass

    def _run_epoch(self, loader, training=True):
        pass

    def fit(self, train_loader, val_loader, epochs):
        pass
