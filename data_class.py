import pandas as pd
from torch.utils.data import Dataset

class CellTypeDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        label = self.labels.iloc[idx,:].to_numpy().astype(float)
        sample = self.samples.iloc[idx,:].to_numpy().astype(float)
        return sample,label

