import os
import luigi
import joblib
import json
import kornia
from tqdm import tqdm
import mlflow
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from preprocess import PreProcess

def replace_relu_with_leakyrelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(negative_slope=0.01))
        else:
            replace_relu_with_leakyrelu(child)

class Train(luigi.Task):
    
    epochs = luigi.IntParameter(default=10)
    train_path = luigi.Parameter(default='intermediates/train.pt')
    test_path = luigi.Parameter(default='intermediates/test.pt')
    encoder_path = luigi.Parameter(default='intermediates/label_encoder.joblib')
    lr = luigi.FloatParameter(default=0.001)
    model = luigi.Parameter(default='ResNet18')
    act_func = luigi.Parameter(default='None')
    optimizer = luigi.Parameter(default='Adam')
    scheduler = luigi.Parameter(default='StepLR')
    loss_fn = luigi.Parameter(default='Cross_Entropy')
    model_weights_path = luigi.Parameter(default='model_weights')
    called = luigi.Parameter(default='None')
    run_name = luigi.Parameter(default='default_experiment')
    data_augment = luigi.Parameter(default='augment')
    save = luigi.Parameter(default='save')

    def requires(self):
        return PreProcess()
    
    def outputs(self):
        return luigi.LocalTarget(os.path.join(self.model_weights_path, self.model + '.pt'))

    def get_paramters_dict(self, is_not=[]):
        params_dict = {}
        for param_name, param_obj in self.param_kwargs.items():
            if param_name in is_not:
                continue 
            params_dict[param_name] = param_obj
        return params_dict

    def run(self):
        print("Started")
        current_uri = mlflow.get_tracking_uri()
        if current_uri is None or current_uri == 'file:///C:/Users/elmow/Documents/Data%20Science/Dog_Classification/mlruns': #resolve this hardcoded section
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        print(mlflow.get_tracking_uri())

        if self.called == 'None':
            kwargs = {'run_name': self.run_name}
        else:
            kwargs = {
                'nested': True
            }

        with mlflow.start_run(**kwargs):
            #Setting up experiment
            mlflow.log_params(self.get_paramters_dict())
            device = ('cuda' if torch.cuda.is_available() else 'cpu') #parameterized this later

            #Loading the data
            loaded_train = torch.load(self.train_path)
            loaded_test = torch.load(self.test_path)
            
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

            if (self.data_augment=='augment'):
                augmentation = kornia.augmentation.AugmentationSequential(
                    kornia.augmentation.RandomHorizontalFlip(),
                    kornia.augmentation.RandomAffine(degrees=10.0, translate=(0.1, 0.1), scale=(0.9, 1.2)),
                    kornia.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    #kornia.color.AdjustContrast(0.1),
                    kornia.enhance.Normalize(mean=mean, std=std),  # Include normalization in the pipeline
                    data_keys=["input"]
                )
                normalize = kornia.enhance.Normalize(mean=mean, std=std)
                augmentation = augmentation.to(device)
            else:
                augmentation=None
                normalize=None

            train_img, train_lbl = loaded_train
            test_img, test_lbl = loaded_test
            train_set = TensorDataset(train_img, train_lbl)
            test_set = TensorDataset(test_img, test_lbl)

            le = joblib.load(self.encoder_path)
            num_classes = len(le.classes_)
           
            
            #Instantiating the training Objects
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=True)

            match self.model:
                case 'ResNet18':
                    model = torchvision.models.resnet18(
                        pretrained=True
                    )
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    model.to(device)
                case 'ResNet34':
                    model = torchvision.models.resnet34(
                        pretrained=True
                    )
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    model.to(device)
                case 'ResNet50':
                    model = torchvision.models.resnet50(
                        pretrained=True
                    )
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    model.to(device)

                case 'ViT16':
                   model = torchvision.models.vit_b_16(weights='DEFAULT')
                   model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
                   model.to(device)
                case _:
                    raise ValueError(f'Unsupported model: {self.model}')
            match self.act_func:
                case 'ReLU': 
                    replace_relu_with_leakyrelu(model)

            match self.optimizer:
                case 'SGD':
                    opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
                case 'Adam':
                    opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
                case 'AdamW':
                    opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-1)
                case _:
                    raise ValueError(f'Unsupported optimizer: {self.optimizer}')
            
            match self.scheduler:
                case 'StepLR':
                    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
                case 'None':
                    scheduler = None
                case _:
                    raise ValueError(f'Unsupported scheduler: {self.scheduler}')
            match self.loss_fn:
                case 'Cross_Entropy':
                    loss_fn = torch.nn.CrossEntropyLoss()
                case _:
                    raise ValueError(f'Unsupported loss function: {self.loss_fn}')

            print(f'Start Training, using {device}')
            #Trainning Loop
            for epoch in range(self.epochs):
                train_loss = 0.0
                train_count = 0.0
                model.train()
                for image, label in tqdm(train_loader, desc='train'):
                    image, label = image.to(device), label.long().to(device)

                    if augmentation != None:
                        image = augmentation(image)

                    opt.zero_grad()
                    output = model(image)

                    loss = loss_fn(output, label)
                    loss.backward()
                    opt.step()
                    

                    train_loss += loss.item()
                    train_count += 1
                if scheduler != None:
                    scheduler.step()
                model.eval()
                test_loss = 0.0
                test_count = 0
                with torch.no_grad():
                    for image, label in tqdm(test_loader, desc='test'):
                        image, label = image.to(device), label.long().to(device)

                        if augmentation != None:
                            image = normalize(image)

                        output = model(image)
                        loss = loss_fn(output, label)

                        test_loss += loss.item()
                        test_count += 1
                
                avg_train_loss = train_loss/train_count
                avg_test_loss = test_loss/test_count

                metrics = {
                    'Train Loss': avg_train_loss,
                    'Test Loss': avg_test_loss
                }
                for param_name in metrics:
                    mlflow.log_metric(param_name, metrics[param_name], step=epoch)
                print(f'Epoch: {epoch}, Train Loss: {avg_train_loss}, Test Loss:{avg_test_loss}')
            if self.save == 'save':
                os.makedirs(self.model_weights_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(self.model_weights_path, self.model + '.pt'))
                
if __name__ == '__main__':
    luigi.build([Train(
        run_name='StepLR_SGD',
        epochs=30,
        model='ResNet34',
        act_func='ReLU',
        data_augment='augment',
        optimizer='SGD',
        scheduler='StepLR',
        lr=0.001)],
        local_scheduler=True
    )
    # luigi.build([Train(
    #     run_name='None',
    #     epochs=40,
    #     model='ResNet18',
    #     data_augment='augment',
    #     optimizer='Adam',
    #     scheduler='StepLR',
    #     lr=0.001)],
    #     local_scheduler=True
    # )


