import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os


def bin_dates(date):

    if date < 1350:
        return "pre 1350"
    elif date < 1450:
        return "1350 - 1450"
    elif date < 1550:
        return "1450 - 1550"
    elif date < 1650:
        return "1550 - 1650"
    elif date < 1750:
        return "1650 - 1750"
    elif date < 1850:
        return "1750 - 1850"
    elif date < 1950:
        return "1850 - 1950"
    elif date < 2000:
        return "1950 - 2000"
    else:
        return "post 2000"


def compute_weights(dataset, column):
    class_weights = compute_class_weight("balanced", classes=np.unique(dataset.data[column]), y=dataset.data[column])
    return torch.tensor(class_weights, dtype=torch.float)



DEFAULT_TRANSFORMER_PIPE = v2.Compose([
    v2.Resize((224,224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class PaintingsDataset(Dataset):
    def __init__(self, csv_file, data_folder, transform=None):
        self.data = pd.read_csv(csv_file)  # CSV with columns: ['image_path', 'style', 'date', 'type']
        self.transform = transform
        self.data_folder = data_folder

        self.data['date_bin'] = self.data['date'].apply(bin_dates)

        self.data["style_encoded"] = LabelEncoder().fit_transform(self.data["style"])
        self.data["date_encoded"] = LabelEncoder().fit_transform(self.data['date_bin'])
        self.data["type_encoded"] = LabelEncoder().fit_transform(self.data["type"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path =  os.path.join(self.data_folder, f"{self.data.iloc[idx, 0]}.jpg",)
        image = read_image(img_path)
        labels = {
            "style": self.data.iloc[idx]["style_encoded"],
            "date": self.data.iloc[idx]["date_encoded"],
            "type": self.data.iloc[idx]["type_encoded"],
        }
        if self.transform:
            image = self.transform(image)
        return image, labels



def CreatePaintingsDataLoaders(csv_file, data_folder, batch_size=32, transformer_pipe=None,):

    

    transformer_pipe = transformer_pipe or DEFAULT_TRANSFORMER_PIPE

    dataset = PaintingsDataset(csv_file=csv_file,data_folder = data_folder,  transform=transformer_pipe)

    val_dataset = PaintingsDataset(csv_file=csv_file, data_folder = data_folder, transform=transformer_pipe)

        
    style_weights = compute_weights(dataset, "style")
    date_weights = compute_weights(dataset, "date_bin")
    type_weights = compute_weights(dataset, "type")

    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,)
    return train_loader, val_loader,[style_weights,date_weights,type_weights],[dataset.data["style"].unique(),dataset.data["date_bin"].unique(),dataset.data["type"].unique()]