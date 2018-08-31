import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

print sys.argv[1]
mypath = sys.argv[1]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print onlyfiles

plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Accuracy')
for file in onlyfiles:
  print mypath+file
  # temp_title = file.split('-')[-1].split('.')[0].split('_')
  temp_title = sys.argv[2]
  plt.title(temp_title)
  df = pd.read_csv(mypath+file)
  if 'uniform' in file:
    label = 'uniform'
    win = 10
  elif 'exp3' in file and 'abs' in file:
    label = 'exp3 (abs)'
    win = 50
  elif 'exp3' in file:
    label = 'exp3'
    win = 50
  elif 'reinforce' in file:
    label = 'reinforce'
    win = 50
  df['Rolled'] = df['Value'].rolling(win).mean()
  plt.plot(df['Step'], df['Rolled'], label=label)
plt.legend()
plt.savefig(mypath+sys.argv[2]+'.png')

print pd.__version__