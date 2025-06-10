from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torchvision.transforms import v2
import os
from utils.data import load_data_module
from utils.sam import *
from utils.eval import *
from models.resnet import *
import wandb
import torch.optim as optim
from tqdm import tqdm
from utils.paths import *
import timm
from transformers import ViTImageProcessor, ViTForImageClassification
import time


class ResNetTrainer():
    def __init__(self, model, batch_size,  opt, lr):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model(True)
        model = model.to(self.device, dtype=torch.float)
        self.model = model

        self.train_data = DataLoader(GTAVDenoisingDataset(True), batch_size=batch_size, shuffle=True)
        self.test_data = DataLoader(GTAVDenoisingDataset(False), batch_size=batch_size, shuffle=True)

        self.optimizer = opt(model.parameters(), lr)

    def train(self, epochs):
        loss_history = torch.zeros((5))
        lh = []
        start = time.time()
        for epoch in range(epochs):
            print('Epoch: ', epoch+1)
            pbar = tqdm(self.train_data)
            for e, (x, y) in enumerate(pbar, 1):
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)
                pred = self.model(x)
                loss = F.mse_loss(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n = min(e,5)
                # .item() takes the single tensor value and makes it a python native element
                loss_history[e%5] = loss.item()
                lh.append(loss.item())
                # this line will print the rolling loss average on the progress bar
                pbar.set_postfix(Loss = (loss_history.sum()/n).item())
        print('Total Training Time: ', time.time() - start)
        return lh

    @torch.inference_mode()
    def test(self):
        # This function calculates the mean average distance of the pixels
        ### and subtracts it from 1 to get an accuracy
        def noiseAccuracy(predicted, target):
            size = target.flatten().shape[0]
            true = (predicted - target).abs().mean()
            return 1-true
        total_accuracy = 0
        start = time.time()
        pbar = tqdm(self.test_data)
        total_steps = 0

        for e, (x, y) in enumerate(pbar, 1):
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float)
            pred = self.model(x)

            # This is the accuracy of the model
            accuracy = noiseAccuracy(pred, y).item()
            total_accuracy += accuracy
            total_steps = e

            pbar.set_postfix(Accuracy = total_accuracy/total_steps)

        print('Noise Accuracy: ' + str((total_accuracy/(total_steps-1))))
        print('Total Eval Time: ', time.time() - start)
