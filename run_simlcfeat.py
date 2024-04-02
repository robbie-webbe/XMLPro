#!/usr/bin/env python3

import sys
import numpy as np
from tqdm import tqdm
from lc_feature import lc_feats

infile = str(sys.argv[1])
dt = float(infile.split('/')[-1].split('_')[-1][2:-4])
data = np.loadtxt(infile, delimiter=',')
feature_array = np.zeros((len(data)-1,42))
for i in tqdm(range(len(data)-1)):
    feature_array[i] = lc_feats(data[0],data[i+1],dt)
np.savetxt('feature_extraction/'+infile.split('/')[-1][:-4]+'_feats.csv',feature_array,delimiter=',')