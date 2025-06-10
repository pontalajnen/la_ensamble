import pickle
import torch
import torchvision
import torchvision.transforms as transforms


class Cifar(torch.utils.data.Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch_size = 4

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

        self.classes = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        sample = (read_image(f'./GTAV/noiseimages/{name}')/255., read_image(f'./GTAV/images/{name}')/255.)
        return sample

    def get_plottable(self, idx):
        name = self.file_names[idx]
        sample = read_image(f'./GTAV/noiseimages/{name}').permute(1,2,0), read_image(f'./GTAV/images/{name}').permute(1,2,0)
        return sample

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
