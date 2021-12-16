import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import math
from typing import Tuple
from utils import progress_bar

from model import LTAE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import copy
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scheduler import CosineWithRestarts

file_list = glob.glob('./data_csvs/*.csv')
file_list = sorted(file_list)
X = []
Y = []
for file in file_list:
  df = pd.read_csv(file,header=None,dtype = str)
  df.dropna(inplace=True)
  # df.replace(['High','Low'],[430,40])
  # df.replace({'High':430,'Low':40},inplace=True)
  arr = df.values
  arr = arr.reshape(-1)
  arr = np.where(arr == 'High', '450', arr)
  arr = np.where(arr == 'Low', '20', arr)
  arr = arr.astype(np.float)
  print(arr.shape[0])
  for i in range(arr.shape[0]):
    if i+54 < arr.shape[0]:
      X.append(arr[i:i+48])
      Y.append(arr[i+54])
  print(np.amax(arr))
  # print(np.argmax(arr))
  # print(arr[np.argmax(arr)])
X = np.array(X)
Y = np.array(Y)
X = X/450
Y = Y/450
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = True, random_state = 8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train = torch.Tensor(X_train) # transform to torch tensor
y_train = torch.Tensor(y_train)

X_test = torch.Tensor(X_test) # transform to torch tensor
y_test = torch.Tensor(y_test)

train_dataset = TensorDataset(X_train,y_train) # create your datset
test_dataset = TensorDataset(X_test,y_test)
trainloader = DataLoader(train_dataset,batch_size=500, shuffle=False, num_workers=1) # create your dataloader
testloader = DataLoader(test_dataset,batch_size=500, shuffle=False, num_workers=1)

# net = Encoder(1,1,1,0.1)
net = LTAE()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = CosineWithRestarts(optimizer, T_max=10)
loss_func = nn.MSELoss()
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape,targets.shape)
        optimizer.zero_grad()
        inputs = torch.unsqueeze(inputs, dim=2)
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        # loss = criterion(outputs, targets)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # predicted = outputs.data.max(1, keepdim=True)[1]
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # correct += predicted.eq(targets.data.max(1, keepdim=True)[1]).sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f '% (train_loss/(batch_idx+1)))
    # wandb.log({"train_loss": train_loss/(batch_idx+1),
    #             "train_accuracy":100.*correct/total,
    #             "epoch":epoch})
    # if (epoch+1)%10 == 0:
    #     print('Saving model..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': 100.*correct/total,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir(args.run_name):
    #         os.makedirs(args.run_name)
    #     torch.save(state, f'./{args.run_name}/ckpt_last.pth')


def test(epoch):
    global least_err
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.unsqueeze(inputs, dim=2)
            outputs = net(inputs)
            outputs = torch.squeeze(outputs)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f '% (test_loss/(batch_idx+1)))
#     # Save checkpoint.
#     acc = 100.*correct/total
#     wandb.log({"test_loss": test_loss/(batch_idx+1),
#             "test_accuracy":acc,
#             "epoch":epoch})
#     if acc > best_acc:
#         print('Saving best model..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir(args.run_name):
#             os.makedirs(args.run_name)
#         torch.save(state, f'./{args.run_name}/ckpt_best.pth')
#         best_acc = acc

if __name__ == "__main__":
    for epoch in range(0, 10):
        train(epoch)
        test(epoch)
        scheduler.step()