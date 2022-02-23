import numpy as np
import matplotlib.pyplot as plt
from ecg_preprocess import ecg_preprocess
from ecg_preprocess import icg_preprocess
from points_detection import points
from scipy.io import loadmat


class data_analysis():
    def __init__(self, files, files_annot, nr, fs, lim, order_highECG, order_lowECG, cutoff_lowECG, cutoff_highECG, order_highICG, order_lowICG, cutoff_lowICG, cutoff_highICG):
        self.files = files
        self.files_annot = files_annot
        self.nr = nr
        self.fs = fs
        self.lim = lim
        self.cutoff_lowECG = cutoff_lowECG
        self.cutoff_highECG = cutoff_highECG
        self.order_highECG = order_highECG
        self.order_lowECG = order_lowECG
        self.cutoff_lowICG = cutoff_lowICG
        self.cutoff_highICG = cutoff_highICG
        self.order_highICG = order_highICG
        self.order_lowICG = order_lowICG

    def Record_analysis(self, plot=False):
        ecg = ecg_preprocess(self.files[self.nr], self.lim, sampling_rate=self.fs, cutoff_low=self.cutoff_lowECG, cutoff_high=self.cutoff_highECG,
                             order_low=self.order_lowECG, order_high=self.order_highECG)
        data_ecg = ecg.sg_filter()
        data_ecg = data_ecg.reshape(data_ecg.shape[0])

        icg = icg_preprocess(self.files[self.nr], self.lim, sampling_rate=self.fs, cutoff_low=self.cutoff_lowICG, cutoff_high=self.cutoff_highICG,
                             order_low=self.order_lowICG, order_high=self.order_highICG)
        data_icg = icg.baseline()

        annotations = loadmat(self.files_annot[self.nr])
        B_ref = annotations['annotPoints'][:, 0]
        C_ref = annotations['annotPoints'][:, 1]
        X_ref = annotations['annotPoints'][:, 2]

        pt = points(data_ecg, data_icg, self.fs)

        R_points = pt.R_peak_detection()
        C_points = pt.C_point_detection()
        T_points = pt.T_point_detection()
        T_end = pt.T_end()
        X_points = pt.X_point_detection()
        B_points = pt.B_point_detection()

        if (plot):
            self.plot_segments(data_icg)
        acc_C = self.accuracy(C_points, C_ref)
        acc_B = self.accuracy(B_points, B_ref)
        acc_X = self.accuracy(X_points, X_ref)
        return acc_C, acc_B, acc_X

    def plotICGECG(self, data_icg, data_ecg):
        fig, axs = plt.subplots(2)

        axs[0].plot(np.arange(len(self.data_icg)), self.data_icg)
        axs[1].plot(np.arange(len(self.data_ecg)), self.data_ecg)
        plt.show()

    def __points_update(self, points, lim_down, lim_up, n=2000):
        points_segment = points[(points >= lim_down) & (points < lim_up)]
        points_segment = points_segment % 2000
        return points_segment


    def plot_segments(self, C_points, B_points, X_points, C_ref, B_ref, X_ref,  n=2000):
        n_segments = int(len(self.data_icg) / n)

        for i in range(n_segments):
            print("i={}".format(i))
            lim_down = i * n
            lim_up = (i + 1) * n
            data_segment = self.data_icg[lim_down: lim_up]

            C_segment = self.points_update(C_points, lim_down, lim_up)
            B_segment = self.points_update(B_points, lim_down, lim_up)
            X_segment = self.points_update(X_points, lim_down, lim_up)

            C_ref_segment = self.points_update(C_ref, lim_down, lim_up)
            B_ref_segment = self.points_update(B_ref, lim_down, lim_up)
            X_ref_segment = self.points_update(X_ref, lim_down, lim_up)

            plt.plot(np.arange(n), data_segment)
            plt.scatter(C_segment, data_segment[C_segment])
            plt.scatter(B_segment, data_segment[B_segment])
            plt.scatter(X_segment, data_segment[X_segment])

            plt.scatter(C_ref_segment, data_segment[C_ref_segment], s=80, facecolors='none', edgecolors='tab:blue')
            plt.scatter(B_ref_segment, data_segment[B_ref_segment], s=80, facecolors='none', edgecolors='tab:orange')
            plt.scatter(X_ref_segment, data_segment[X_ref_segment], s=80, facecolors='none', edgecolors='tab:green')

            plt.show()

    def accuracy(self, points, points_ref):
        errors = []
        for i in range(len(points_ref)):
            p_ref = points_ref[i]
            error = np.min(abs(points - p_ref))
            errors.append(error)
        return np.mean(errors)


