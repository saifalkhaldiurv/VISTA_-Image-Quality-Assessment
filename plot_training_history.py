#%%
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
  ax2.plot(history['test_acc'], label='test accuracy')
  ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.set_ylim([-0.05, 1.05])
  ax2.legend()
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  fig.suptitle('Training history')
  plt.show()

history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': [], 'test_acc': []}

with open('titan/nohup.out', 'r', encoding='utf-8') as f:
    for line in f:
        if 'Train loss' in line:
            train_loss = float(re.search(r'Train loss ([\d.]+)', line).group(1))
            train_acc = float(re.search(r'accuracy ([\d.]+)', line).group(1))
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
        elif 'validation   loss' in line:
            val_loss = float(re.search(r'validation   loss ([\d.]+)', line).group(1))
            val_acc = float(re.search(r'accuracy ([\d.]+)', line).group(1))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        elif 'Test accuracy' in line:
            test_acc = float(re.search(r'Test accuracy ([\d.]+)', line).group(1))
            history['test_acc'].append(test_acc)

plot_training_history(history)

#%%
import pandas as pd

df = pd.read_csv('titan/test.csv')

classes = df['correct'].unique()
accuracy = {}

for c in classes:
    class_df = df[df['correct'] == c]
    correct = class_df[class_df['correct'] == class_df['predict']]
    accuracy[c] = len(correct) / len(class_df)

print(accuracy)

#%%
