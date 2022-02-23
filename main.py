import numpy as np
from glob import glob
import re
from ecg_preprocess import ecg_preprocess
from ecg_preprocess import icg_preprocess
from points_detection import points
from scipy.io import loadmat

from data_analysis import data_analysis

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]



# 1. DATA LOAD

files = glob("01_RawData/*BL.mat")
files.sort(key=natural_keys)

files_annot = glob("03_ExpertAnnotations/*BL.mat")
files_annot.sort(key=natural_keys)


# 2. DATA PARAMETERS

lim = -1
fs = 500

# a) ECG parameters

cutoff_lowECG = 17
order_lowECG = 4
cutoff_highECG = 0.5
order_highECG = 4

# b) ICG parameters

cutoff_lowICG = 20
order_lowICG = 4
cutoff_highICG = 5
order_highICG = 4

da = data_analysis(files, files_annot, 0, fs, lim, order_highECG, order_lowECG, cutoff_lowECG, cutoff_highECG, order_highICG, order_lowICG, cutoff_lowICG, cutoff_highICG)
acc_C, acc_B, acc_X = da.Record_analysis()

print("Accuracy C_point:{}".format(acc_C*2))
print("Accuracy B_point:{}".format(acc_B*2))
print("Accuracy X_point:{}".format(acc_X*2))



