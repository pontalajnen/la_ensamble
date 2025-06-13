import torch
import torchvision
import torchvision.transforms as transforms


class Cifar(torch.utils.data.Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch_size = 128

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        self.classes = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.train_loader.dataset) + len(self.test_loader.dataset)

    def __getitem__(self, idx):
        if idx < len(self.train_loader.dataset):
            return self.train_loader.dataset[idx]
        else:
            return self.test_loader.dataset[idx - len(self.train_loader.dataset)]
