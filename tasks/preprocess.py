import os
import luigi
import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class EnsureDirectoryExists(luigi.Task):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        os.makedirs(self.path, exist_ok=True)
    
class CustomTensorDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.data = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.le = LabelEncoder()
        self.le.fit(self.data.breed.unique())
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, "id"] + ".jpg")
        image = Image.open(img_name)
        label_int = torch.tensor(self.le.transform([self.data.loc[idx, "breed"]]), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label_int

class PreProcess(luigi.Task):

    train_path = luigi.Parameter(default='intermediates/train.pt')
    test_path = luigi.Parameter(default='intermediates/test.pt')
    train_data = luigi.Parameter(default='data/train')
    labels_csv = luigi.Parameter(default='data/labels.csv')
    encoder_path = luigi.Parameter(default='intermediates/label_encoder.joblib')
    train_split = luigi.FloatParameter(default=0.8)

    def require(self):
        return [
            EnsureDirectoryExists(self.train_data),
            EnsureDirectoryExists(self.submission_data),
            EnsureDirectoryExists(self.labels_csv)
        ]

    def output(self):
        return {
            'train':luigi.LocalTarget(self.train_path),
            'test': luigi.LocalTarget(self.test_path),
            'encoder': luigi.LocalTarget(self.encoder_path)
        }

    def run(self):

        labels = pd.read_csv(self.labels_csv)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet normalization
        ])

        data_set = CustomTensorDataset(self.train_data, labels, transform=transform)
        
        train_len = int(self.train_split * len(data_set))
        test_len = int(len(data_set) - train_len)
        train_set, test_set = random_split(data_set, [train_len, test_len])

        channels, height, width = 3, 224, 224
        train_image = torch.empty((train_len, channels, height, width))
        train_label = torch.zeros(train_len)
        test_image = torch.empty((test_len, channels, height, width))
        test_label = torch.zeros(test_len)

        for i, (image, label) in enumerate(train_set):
            train_image[i] = image
            train_label[i] = label

        for i, (image, label) in enumerate(test_set):
            test_image[i] = image
            test_label[i] = label

        os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
        torch.save([train_image, train_label], self.train_path)
        torch.save([test_image, test_label], self.test_path)
        joblib.dump(data_set.le, self.encoder_path)

if __name__ == '__main__':
    luigi.build([PreProcess()], local_scheduler=True)
