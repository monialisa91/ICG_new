import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from ecg_preprocess import ecg_preprocess
from ecg_preprocess import icg_preprocess
from points_detection import points
from qrs_detection import qrs

# POPRAWI WYZNACZANIE QRS

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def plotICGECG(data_icg, data_ecg):
    fig, axs = plt.subplots(2)

    axs[0].plot(np.arange(len(data_icg)), data_icg)
    axs[1].plot(np.arange(len(data_ecg)), data_ecg)
    plt.show()


# 1. DATA LOAD

files = glob("01_RawData/*BL.mat")
files.sort(key=natural_keys)

# 2. DATA PARAMETERS

lim = 1000
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


# 3. LOAD OF THE PREPROCESSED FILES

ecg = ecg_preprocess(files[0], lim, sampling_rate=fs, cutoff_low=cutoff_lowECG, cutoff_high=cutoff_highECG, order_low=order_lowECG, order_high=order_highECG)
data_ecg = ecg.sg_filter()
data_ecg = data_ecg.reshape(data_ecg.shape[0])

qrs = qrs(data_ecg, fs)
new_data = qrs.enhancement_mask()
points = qrs.crest_and_troughs()
plt.plot(np.arange(len(new_data)), data_ecg*1000)
plt.plot(np.arange(len(new_data)), new_data)
plt.scatter(points, new_data[points], color="tab:red")
plt.show()
exit()


icg = icg_preprocess(files[0], lim, sampling_rate=fs, cutoff_low=cutoff_lowICG, cutoff_high=cutoff_highICG, order_low=order_lowICG, order_high=order_highICG)
data_icg = icg.baseline()

fd = np.gradient(data_icg)
sd = np.gradient(fd)

pt = points(data_ecg, data_icg, fs)

R_points = pt.R_peak_detection()
C_points = pt.C_point_detection()
T_points = pt.T_point_detection()
T_end = pt.T_end()
S_points = pt.S_point_detection()
X_points = pt.X_point_detection()
B_points = pt.B_point_detection()

fig, ax = plt.subplots(3)

ax[0].plot(np.arange(len(data_ecg)), data_ecg)

# ax[0]. scatter(R_points, data_ecg[R_points])
# ax[0].scatter(T_points, data_ecg[T_points])
# ax[0].scatter(T_end, data_ecg[T_end])
#
# ax[0].scatter(S_points, data_ecg[S_points])
#
#
ax[1].plot(np.arange(len(data_icg)), data_icg)
ax[1].scatter(C_points, data_icg[C_points])
ax[1].scatter(X_points, data_icg[X_points])
ax[1].scatter(B_points, data_icg[B_points])

ax[2].plot(np.arange(len(data_icg)), sd)
ax[2].scatter(B_points, sd[B_points])


plt.show()

