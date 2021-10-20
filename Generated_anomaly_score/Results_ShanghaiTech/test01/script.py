import numpy as np
from sklearn import metrics
import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def check(i,curr):
    for x in curr:
        if in_range(i,x):
            return True
    return False

def in_range(i,x):
    return int(x[0]) <= i <= int(x[1])

output_files = sorted(glob.glob('*.csv'))
cum_auc = 0

with open('observe01') as f:
    lines = [line.rstrip('\n') for line in f]

# r is a list(list)
# each element of r is of form ['5:90', ' 140:200']
r=[x.split(',') for x in lines]

# l is a list(list(list))
# each element of val is of form [['5', '90'], [' 140', '200']]
val = list(map(lambda xs: [x.split(':') for x in xs],r))

with open('results/results.txt','w') as f:
    for file_num,_file in enumerate(output_files):
        data = pd.read_csv(_file)


        scores = np.array(data['score'])
        #m = min(scores); M = max(scores) 
        #scores = np.array([(x-m)/(M-m) for x in scores])

        curr = val[file_num]

        true_values = []

        for i in range(len(scores)):
            if check(i,curr):
                true_values.append(0)
            else:
                true_values.append(1)
        y_true = np.array(true_values)

        fpr,tpr,thresholds = metrics.roc_curve(y_true,scores)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        auc = metrics.roc_auc_score(y_true,scores)
        f.write('auc = ' + str(auc) + ', eer = ' + str(eer) + '\n')
        #thresh = interp1d(fpr, thresholds)(eer)
        cum_auc += auc

        plt.figure()
        plt.plot(fpr,tpr)
        
        n = file_num+1 
        plt.savefig(f"results/roc_{n}.pdf")

print(cum_auc/len(output_files))

