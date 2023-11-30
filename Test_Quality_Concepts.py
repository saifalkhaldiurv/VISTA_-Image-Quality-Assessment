from __future__ import print_function,division,absolute_import
# import visionmaster
# from visionmaster.densenet import *
import os
import pathlib
import cv2
import shutil
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import seaborn as sns
from pathlib import Path
import torch, torchvision
from PIL.Image import Image
import torch.autograd
from matplotlib import rc
from pylab import rcParams
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator
from torchvision.datasets import ImageFolder
from torchvision import datasets,models,transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.utils import save_image
import pandas as pd
from models.vgg16_patch import UNet16
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from models.loss import DiceBCELoss,DiceLoss,TverskyLoss
from Test_Quality import create_model
import argparse

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')

def train_epoch(model,dataloaders,loss_fn,loss_MSE,loss_DiceBCE,optimizer,device,scheduler,n_examples,concept_loaders):
  model = model.train()
  losses = []
  correct_predictions = 0
  i = 1
  for inputs, labels in tqdm(dataloaders):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # update the CW parameters, not used when training standard network
    if (i + 1) % 30 == 0:
        model.eval()
        with torch.no_grad():
            # update the gradient matrix G
            for concept_index, concept_loader in enumerate(concept_loaders):
                model.change_mode(concept_index)
                for j, (X, _) in enumerate(concept_loader):
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break
            model.update_rotation_matrix()
            # change to ordinary mode
            model.change_mode(-1)
        model.train()

    recoimage, outputs = model(inputs)
    #print("################",outputs.shape)

    _,preds = torch.max(outputs, dim=1)
    loss1 = loss_fn(outputs,labels)
    #loss2 = ssim(inputs, recoimage) 
    # print("====================================")
    # print(inputs.shape)
    # print(recoimage.shape)
    loss2 = loss_MSE(inputs, recoimage)
    loss3 = loss_DiceBCE(inputs, recoimage)
    
    if i%100 == 0:
        os.makedirs('s', exist_ok=True)
        save_image(inputs, 's/'+str(i)+'_Input.jpeg')
        save_image(recoimage, 's/'+str(i)+'_Reco.jpeg')
    i +=1

    loss = (loss1+loss2+loss3)/3
    # loss = loss1
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  return model, correct_predictions.double() / n_examples ,np.mean(losses) # 
# ================================================================ # 
def eval_model(model, dataloaders, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      _, outputs = model(inputs)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses) 

#===========================================================================#
def test_model(model, dataloaders, device, n_examples):
  model = model.eval()
  df = pd.DataFrame(columns=["No","correct","predict"])
  i=1
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      recoimage, outputs = model(inputs)
      if i%100 == 0:
        os.makedirs('Test', exist_ok=True)
        save_image(inputs, 'Test/'+str(i)+'_Input.jpeg')
        save_image(recoimage, 'Test/'+str(i)+'_Reco.jpeg')
      _, preds = torch.max(outputs, dim=1)
      correct_predictions += torch.sum(preds == labels)
      df.loc[i] =  [i,labels.data.cpu().numpy()[0],preds.data.cpu().numpy()[0]]
      i +=1

  os.makedirs('result', exist_ok=True)
  df.to_csv("result/test.csv", sep=',',index=False)
  return correct_predictions.double() / n_examples
#-----------------------------------------------------------
# ================================================================ # 
def checkpoint_path(filename,model_name):
  checkpoint_folderpath = pathlib.Path(f'checkpoint/{model_name}')
  checkpoint_folderpath.mkdir(exist_ok=True,parents=True)
  return checkpoint_folderpath/filename

def directories_to_list(path):
  directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
  return directories

def train_model(model, dataloaders, dataloaders_test, dataset_sizes, device, n_epochs=50):
  concept_loaders = [
      torch.utils.data.DataLoader(
      datasets.ImageFolder(os.path.join("data_ls/concept_train", concept), transforms.Compose([
        #   transforms.RandomResizedCrop(224),
        #   transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
      ])),
      batch_size=1, shuffle=True,
      num_workers=4, pin_memory=False)
      for concept in directories_to_list("data_ls/concept_train")
  ]
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  loss_fn = nn.CrossEntropyLoss(reduction='mean').to(device)
  loss_MSE = nn.MSELoss().to(device)
  loss_DiceBCE= TverskyLoss().to(device)
  best_model_path = checkpoint_path('best_model_state.ckpt',model.name)
  # model.load_state_dict(torch.load(best_model_path))
  model.eval()
  #print(model)  
  history = defaultdict(list)
  best_accuracy = 0
  for epoch in range(last_epoch+1-10, n_epochs):
    epoch = epoch + 10
    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)
    adjust_learning_rate(optimizer, epoch)
    model, train_acc, train_loss = train_epoch(model,dataloaders['train'],loss_fn,loss_MSE,loss_DiceBCE,optimizer,device,scheduler,dataset_sizes['train'],concept_loaders)

    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model,dataloaders['val'],loss_fn,device,dataset_sizes['val'])
    print(f'validation   loss {val_loss} accuracy {val_acc}')
    test_acc = test_model(model, dataloaders_test['test'], device, dataset_sizes['test'])
    print(f'Test accuracy {test_acc}')
  
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    torch.save(model.state_dict(), checkpoint_path('best_model_state_'+str(epoch)+'.ckpt',model.name))
    if test_acc > best_accuracy:
      torch.save(model.state_dict(), best_model_path)
      best_accuracy = test_acc
  print(f'Best val accuracy: {best_accuracy}')
  model.load_state_dict(torch.load(best_model_path))
  return model, history
 
def plot_training_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
  ax1.plot(history['train_loss'], label='train loss')
  ax1.plot(history['val_loss'], label='validation loss')
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.set_ylim([-0.05, 1.05])
  ax1.legend()
  ax1.set_ylabel('Loss')
  ax1.set_xlabel('Epoch')
  ax2.plot(history['train_acc'], label='train accuracy')
  ax2.plot(history['val_acc'], label='validation accuracy')
  ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.set_ylim([-0.05, 1.05])
  ax2.legend()
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  fig.suptitle('Training history')
# ================================================================ #

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import os
import re

def get_last_epoch(folder_path: str) -> int:
    pattern = r'best_model_state_(\d+).ckpt'
    last_epoch = 0
    for filename in os.listdir(folder_path):
        match = re.search(pattern, filename)
        if match:
            epoch = int(match.group(1))
            last_epoch = max(last_epoch, epoch)
    return last_epoch

def directories_to_string(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return ','.join(directories)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

if __name__ == '__main__':

  # from roboflow import Roboflow
  # rf = Roboflow(api_key="05kzxjUKzL75iMzd9qy6")
  # project = rf.workspace("fundus-gradable").project("fundus-concepts")
  # dataset = project.version(2).download("multiclass", location="data/concepts", overwrite=False)

  global args
  args = parser.parse_args()
  # ================================================================ # 
  data_dir='data_ls/'#/media/saif/218E2FB45FA456AE/saif data/dataset'    #'/media/saif/218E2FB45FA456AE/saif data/data_Quality_3_ag'
  train_dir=data_dir+'/train'
  valid_dir=data_dir+'/val'
  test_dir=data_dir+'/test'
  # ================================================================ # 
  # Data augmentation and normalization for training
  # Just normalization for validation
  resolution = 1280
  data_transforms = {
      'train': transforms.Compose([
        transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(180),
          transforms.ColorJitter(brightness=0.01,contrast=0.01,hue=0.01,saturation=0.01),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }
  model_name = 'vgg16patch' 
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train','val', 'test']}
  dataloaders= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
                                              shuffle=True, num_workers=4)
                for x in  ['train','val']}
  
  dataloaders_test= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=False, num_workers=2,drop_last=False)
                for x in  ['test']}
              
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val', 'test']}
 

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # ================================================================ # 
  print("Device: ", device)
  class_names = image_datasets['train'].classes
  print("Classes: ", class_names)
  # ================================================================ # 
  base_model, encoder = create_model(model_name,num_classes=len(class_names),device=device)
  # ================================================================ # 

  # base_model = torch.nn.DataParallel(base_model)
  # base_model = base_model.to(device)
  print ("Model: ",model_name)
  print (base_model)
  last_epoch = get_last_epoch('checkpoint/'+model_name)
  print(f'Last epoch: {last_epoch}')
  if last_epoch != 0:
    base_model.load_state_dict(torch.load('checkpoint/'+model_name+f'/best_model_state_{last_epoch}.ckpt',map_location="cuda"))#,strict=False)
  print(directories_to_list("data_ls/concept_train"))

  from plot_functions import *
  class MyObject:
    def __init__(self, data):
        self.__dict__ = data
  data = {"arch": model_name,
          "depth": '16',
          "concepts":directories_to_string('data_ls/concept_train')}
  myobj = MyObject(data)
  dataloaders_test_dir= torch.utils.data.DataLoader(
     ImageFolderWithPaths(os.path.join(data_dir, 'test'), transform=data_transforms['test']), 
     batch_size=2,shuffle=True, num_workers=1,pin_memory=False)
  # print(next(iter(dataloaders_test_dir)))
  plot_concept_top50(myobj, dataloaders_test_dir, base_model, '8')

  import sys
  sys.exit()

  # ================================================================ # 
  base_model, history = train_model(base_model, dataloaders, dataloaders_test, dataset_sizes, device)
  # ================================================================ # 
  plot_training_history(history)
  # ================================================================ # 
