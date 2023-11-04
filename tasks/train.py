import os
import luigi
import joblib
from tqdm import tqdm
import mlflow
import torch
import torchvision
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from preprocess import PreProcess
class Train(luigi.Task):
    
    epochs = luigi.IntParameter(default=10)
    train_path = luigi.Parameter(default='intermediates/train.pt')
    test_path = luigi.Parameter(default='intermediates/test.pt')
    encoder_path = luigi.Parameter(default='intermediates/label_encoder.joblib')
    lr = luigi.FloatParameter(default=0.001)
    model = luigi.Parameter(default='ResNet18')
    optimizer = luigi.Parameter(default='Adam')
    loss_fn = luigi.Parameter(default='Cross_Entropy')
    model_weights_path = luigi.Parameter(default='model_weights')

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
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        with mlflow.start_run(run_name="test"):
            #Setting up experiment
            mlflow.log_params(self.get_paramters_dict())

            #Loading the data
            loaded_train = torch.load(self.train_path)
            loaded_test = torch.load(self.test_path)

            train_img, train_lbl = loaded_train
            test_img, test_lbl = loaded_test
            train_set = TensorDataset(train_img, train_lbl)
            test_set = TensorDataset(test_img, test_lbl)

            le = joblib.load(self.encoder_path)
            num_classes = len(le.classes_)

            #Instantiating the training Objects
            device = ('cuda' if torch.cuda.is_available() else 'cpu') #parameterized this later
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=True)

            match self.model:
                case 'ResNet18':
                    model = torchvision.models.resnet18(
                        pretrained=True
                    )
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    model.to(device)
                case _:
                    raise ValueError(f'Unsupported model: {self.model}')
            
            match self.optimizer:
                case 'Adam':
                    opt = torch.optim.Adam(model.parameters(), lr=self.lr)
                case _:
                    raise ValueError(f'Unsupported optimizer: {self.optimizer}')
            
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
                for image, label in train_loader:
                    image, label = image.to(device), label.long().to(device)

                    opt.zero_grad()
                    output = model(image)

                    loss = loss_fn(output, label)
                    loss.backward()
                    opt.step()

                    train_loss += loss.item()
                    train_count += 1

                model.eval()
                test_loss = 0.0
                test_count = 0
                with torch.no_grad():
                    for image, label in test_loader:
                        image, label = image.to(device), label.long().to(device)

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
                #print(f'Epoch: {epoch}, Train Loss: {avg_train_loss}, Test Loss:{avg_test_loss}')
            os.makedirs(self.model_weights_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(self.model_weights_path, self.model + '.pt'))
                
if __name__ == '__main__':
    luigi.build([Train()], local_scheduler=True)


