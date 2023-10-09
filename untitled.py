import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import subplots

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

root_dir = 'train/'
csv_file = 'labels.csv'

#Hyperparameters
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 64

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.data = csv_file
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.loc[idx, "id"] + ".jpg")
        image = Image.open(img_name)
        label_int = int(self.data.loc[idx, "breed"])
        # one_hot_label = torch.zeros(self.num_classes)
        # one_hot_label[label_int] = 1
        if self.transform:
            image = self.transform(image)
        return image, label_int

labels_df = pd.read_csv(csv_file)
num_classes = labels_df['breed'].nunique()

# Create a label encoder object
le = LabelEncoder()

# Fit and transform the 'breed' column to get integer labels
labels_df['breed'] = le.fit_transform(labels_df['breed'])

#Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet normalization
])

#Datasets
dataset = CustomImageDataset(root_dir=root_dir, csv_file=labels_df, transform=transform)
total_samples = len(dataset)
train_size = int(TRAIN_SIZE * total_samples)
test_size = total_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

Resnet
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

#Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#Using CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Trainning Loop
num_epochs = 10
errors_list = []
for epoch in range(num_epochs):
    
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss_val = criterion(outputs, labels)
        loss_val.backward()
        optimizer.step()

        running_train_loss += loss_val.item()

    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            
            running_test_loss += loss_val.item()

    avg_train_loss = running_train_loss / len(train_loader)
    avg_test_loss = running_test_loss / len(test_loader)
    
    errors_list.append({'test_loss': avg_test_loss, 'train_loss': avg_train_loss})

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss}, Test_Loss: {avg_test_loss}")