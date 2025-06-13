import torch
# from torchvision.transforms import v2
# import os
# from utils.data import load_data_module
# from utils.sam import *
# from utils.eval import *
# from models.resnet import *
# import wandb
# import torch.optim as optim
from tqdm import tqdm
# from utils.paths import *
# import timm
# from transformers import ViTImageProcessor, ViTForImageClassification
import time
# import torchvision
from dataset import Cifar
import torch.nn as nn


class ResNetTrainer():
    def __init__(self, model, batch_size,  optimizer, dataset: Cifar):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        print("Device:", self.device)

        model = model.to(self.device)
        self.model = model

        self.train_data = dataset.train_loader
        self.test_data = dataset.test_loader

        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        loss_history = torch.zeros((5))
        lh = []
        start = time.time()

        for epoch in range(epochs):
            print('Epoch: ', epoch+1)
            progress_bar = tqdm(self.train_data)
            for e, (x, y) in enumerate(progress_bar, 1):
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device)
                prediction = self.model(x)
                loss = self.criterion(prediction, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n = min(e, 5)
                loss_history[e % 5] = loss.item()
                lh.append(loss.item())
                progress_bar.set_postfix(Loss=(loss_history.sum()/n).item())
            torch.save(self.model.state_dict(), f"saved_models/model_epoch_{epoch+1}.pth")
        print('Total Training Time: ', time.time() - start)

        torch.save(self.model.state_dict(), "saved_models/model_final.pth")
        return lh

    @torch.inference_mode()
    def test(self):
        correct = 0
        total = 0
        progress_bar = tqdm(self.test_data)
        self.model.eval()

        start = time.time()
        with torch.no_grad():
            for e, (x, y) in enumerate(progress_bar, 1):
                x = x.to(self.device)
                y = y.to(self.device)
                images, labels = x, y

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy : {100 * correct // total} %')
        print(f"Total Testing Time: {time.time() - start} seconds")
