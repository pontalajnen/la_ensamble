import torch
import pickle


class GTAVDenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.file_names = []
        split = 'train' if train else 'test'
        with open(f'GTAV/{split}.txt', 'r') as f:
            entry = f.readline()
            while entry:
                self.file_names.append(entry.strip())
                entry = f.readline()

    def __len__(self):
        ### returns the number of items in the dataset, nice and simple
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        ### of the form (x, y) the values are scaled by 255 and returned as type float
        sample = (read_image(f'./GTAV/noiseimages/{name}')/255., read_image(f'./GTAV/images/{name}')/255.)
        return sample

    def get_plottable(self, idx):
        ### Same deal as before but this time the images are permuted and not scaled
        name = self.file_names[idx]
        sample = read_image(f'./GTAV/noiseimages/{name}').permute(1,2,0), read_image(f'./GTAV/images/{name}').permute(1,2,0)
        return sample

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
