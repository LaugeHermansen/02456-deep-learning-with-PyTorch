#%%

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from collections import defaultdict
from tools import Timer, glob, mkdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
#import train test split
from sklearn.model_selection import train_test_split

#%%

class MyMNIST(torchvision.datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.index = {i:[] for i in range(10)}
        for idx,target in enumerate(self.targets):
            self.index[target.item()].append(idx)
        for target, idxs in self.index.items():
            self.index[target] = torch.tensor(idxs)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target



class Model(nn.Module):
    def __init__(self, conv_channels, linear_layers, dropout, dimension = (28,28)):
        super().__init__()
        self.log_data = defaultdict(list)
        self.loss_fn = nn.CrossEntropyLoss()
        modules = []
        dimension = np.array(dimension)
        self.register_buffer("epoch", torch.tensor(0))


        for d_in, d_out in zip(conv_channels[:-1], conv_channels[1:]):
            modules.append(nn.BatchNorm2d(d_in))
            modules.append(nn.Conv2d(d_in, d_out, kernel_size=3, padding=1, stride=1))
            modules.append(nn.MaxPool2d(kernel_size=2))
            dimension = (dimension/2).astype(int)
            modules.append(nn.ReLU())
        dimension = np.prod(dimension)*conv_channels[-1]
        assert dimension == linear_layers[0], f"Dimension mismatch: {dimension} != {linear_layers[0]}"
        modules.append(nn.Flatten(1,-1))
        for d_in, d_out in zip(linear_layers[:-1], linear_layers[1:]):
            modules.append(nn.BatchNorm1d(d_in))
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(d_in, d_out))
            modules.append(nn.ReLU())
        self.module_list = nn.ModuleList(modules)

    def __repr__(self):
        t = {label:len(value) for label, value in self.log_data.items() if label != "steps_pr_epoch"}
        t["epoch"] = self.epoch.item()
        return str(t)

    def forward(self, X):
        X = X.unsqueeze(1).float()
        for module in self.module_list:
            X = module(X)
        return X

    def inference(self, X):
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred
    
    def train_step(self, X, y):
        self.train()
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate_step(self, X, y):
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y)
        return loss.item()
    
    def accuracy(self, X, y, mean=True):
        self.eval()
        y_pred = self.inference(X)
        return (y_pred.detach() == y.detach()).sum().item()/(len(y) if mean else 1)

    def fit(self, train_loader, val_loader_step, val_loader_epoch, epochs, lr, log_interval=100,
            time=False, verbose=False, use_tqdm=False):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        timer = Timer(time)

        epoch_iterator = range(self.epoch, epochs + self.epoch) if not use_tqdm else tqdm(range(self.epoch, epochs + self.epoch))
        for epoch in epoch_iterator:

            if verbose: print(f"Epoch: {epoch}")
            timer("train_step")
            for step, (X, y) in enumerate(train_loader):
                loss = self.train_step(X, y)
                self.log_data['train_loss_step'].append(loss)
                self.log_data['train_step_step'].append(step + len(train_loader)*epoch)
                timer("train_acc", stop_previous=False)
                self.log_data['train_acc_step'].append(self.accuracy(X, y))
                timer()
                if step%log_interval == 0 or step == len(train_loader)-1:
                    timer("validate_step")
                    val_loss = 0
                    val_acc = 0
                    for X, y in val_loader_step:
                        val_loss += self.validate_step(X, y)
                        val_acc += self.accuracy(X, y, mean=False)
                    val_loss /= len(val_loader_step)
                    val_acc /= len(val_loader_step.dataset)
                    self.log_data['val_loss_step'].append(val_loss)
                    self.log_data['val_acc_step'].append(val_acc)
                    self.log_data['val_step_step'].append(step + len(train_loader)*epoch)

                    self.log_data['params'].append(self.get_params())
                    if verbose: print(f"Step: {step}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}")
            timer("validate_epoch")
            for X, y in val_loader_epoch:
                val_loss += self.validate_step(X, y)
                val_acc += self.accuracy(X, y, mean=False)
            
            self.epoch += 1

            val_loss /= len(val_loader_epoch)
            val_acc /= len(val_loader_epoch.dataset)
            self.log_data['val_loss_epoch'].append(val_loss)
            self.log_data['val_acc_epoch'].append(val_acc)
            self.log_data['val_step_epoch'].append(len(val_loader_epoch)*(epoch+1))

            if verbose: print(f" ------- Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f} --------")
        
        timer()

        self.log_data['steps_pr_epoch'] = len(train_loader)
        if time: print(timer.evaluate())
        return self.log_data

    def get_params(self):
        return torch.cat([x.view(1,-1) for x in self.module_list.parameters()], dim=1).detach().cpu().numpy().squeeze()

#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



data_fitting = MyMNIST(root='data', train=True, download=True)
data_fitting.data = data_fitting.data.to(device)
data_fitting.targets = data_fitting.targets.to(device)

data_train, data_val_step, data_val_epoch = torch.utils.data.random_split(data_fitting, [49000, 1000, 10000], generator=torch.Generator().manual_seed(42))
# data_train, data_val_step, data_val_epoch, _ = torch.utils.data.random_split(data_fitting, [10000, 100, 1000, 48900], generator=torch.Generator().manual_seed(42))

data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=True)
data_val_step_loader = torch.utils.data.DataLoader(data_val_step, batch_size=100, shuffle=True)
data_val_epoch_loader = torch.utils.data.DataLoader(data_val_epoch, batch_size=100, shuffle=True)

data_test = MyMNIST(root='data', train=False, download=True)
data_test.data = data_test.data.to(device)
data_test.targets = data_test.targets.to(device)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=True)





#%%

model_params = {"conv_channels":[1,4,8,16,32], "linear_layers":[32, 10], "dropout":0.5,}
fit_params = {"epochs":10, "lr":0.02, "log_interval":100, "use_tqdm":True}

model_start = Model(**model_params)
n_leaves = 5
n_experiments = 5
models = []

#%%

model_current_template = Model(**model_params)
model_current_template.load_state_dict(model_start.state_dict())
model_current_template.to(device)



for exp in range(n_experiments):
    print(f"Experiment {exp+1}/{n_experiments}")
    current_models = [model_current_template]
    for leaf in range(n_leaves):
        copy = Model(**model_params)
        copy.load_state_dict(model_current_template.state_dict())
        copy.to(device)
        current_models.append(copy)

    for i in range(len(current_models)): 
        print(f"Leaf {i+1}/{n_leaves+1}")
        current_models[i].fit(data_train_loader, data_val_step_loader, data_val_epoch_loader, **fit_params)

    current_model_idx = np.argmin([model.log_data['val_loss_epoch'][-1] for model in current_models])
    model_current_template = current_models[current_model_idx]

    for i, model in enumerate(current_models):
        if i != current_model_idx or exp == n_experiments-1:
            models.append(model)
        


# #%%

# for model in models:
#     plt.plot(model.log_data['train_step_step'], model.log_data['train_loss_step'], label="train_loss")
#     plt.plot(model.log_data['val_step_step'], model.log_data['val_loss_step'], label="val_loss")
#     plt.legend()
#     plt.show()
#     plt.plot(model.log_data['train_step_step'], model.log_data['train_acc_step'], label="train_acc")
#     plt.plot(model.log_data['val_step_step'], model.log_data['val_acc_step'], label="val_acc")
#     plt.legend()
#     plt.show()


# #%%

# all_params = np.concatenate([model.log_data['params'] for model in models[-6:]], axis=0)

# sc = StandardScaler(with_mean=False)
# all_params_sc = sc.fit_transform(all_params)

# pca = PCA(n_components=10)
# pca.fit(all_params_sc)

# #%%

# for model in models[-6:]:
#     params = model.log_data['params']
#     params_tr = pca.transform(sc.transform(params))
#     plt.scatter(params_tr[:,0], params_tr[:,1], c=model.log_data['val_acc_step'], cmap='viridis', s=1)
#     plt.plot(params_tr[:,0], params_tr[:,1], c='k', alpha=1, linewidth = 0.5)
#     plt.plot(params_tr[-1,0], params_tr[-1,1], c='r', marker='o')
# plt.colorbar()
# # plt.show()
# # plt.plot(np.cumsum(pca.explained_variance_ratio_))

# #%%


# model_avg = Model(**model_params)

# for i in range(len(model_avg.module_list)):
#     if isinstance(model_avg.module_list[i], nn.BatchNorm2d) or isinstance(model_avg.module_list[i], nn.BatchNorm1d):
#         model_avg.module_list[i].running_mean.data = torch.mean(torch.stack([model.module_list[i].running_mean.data for model in models[-30:]]), dim=0)
#         model_avg.module_list[i].running_var.data = torch.mean(torch.stack([model.module_list[i].running_var.data for model in models[-30:]]), dim=0)
#     if isinstance(model_avg.module_list[i], nn.Conv2d) or isinstance(model_avg.module_list[i], nn.Linear) or isinstance(model_avg.module_list[i], nn.BatchNorm2d) or isinstance(model_avg.module_list[i], nn.BatchNorm1d):
#         model_avg.module_list[i].weight.data = torch.mean(torch.stack([model.module_list[i].weight.data for model in models[-30:]]), dim=0)
#         model_avg.module_list[i].bias.data = torch.mean(torch.stack([model.module_list[i].bias.data for model in models[-30:]]), dim=0)


# model_avg.to(device)

# acc = 0
# loss = 0

# for X,y in data_val_epoch_loader:
#     y_pred = model_avg(X)
#     acc += model_avg.accuracy(X, y, False)
#     loss += model_avg.loss_fn(y_pred, y).item()

# acc /= len(data_val_epoch)
# loss /= len(data_val_epoch)

# print(f"Accuracy: {acc}")
# print(f"Loss: {loss}")




