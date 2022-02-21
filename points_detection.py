import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class points():

    def __init__(self, data_ecg, data_icg, fs):
        self.data_ecg = data_ecg
        self.data_icg = data_icg
        self.fs = fs

    def R_peak_detection(self):
        data_pt = self.data_ecg
        peaks = find_peaks(data_pt, distance=150)[0]
        values = data_pt[np.array(peaks)]
        maksimum = np.sort(values)[-2:]
        thr = 0.8 * np.mean(maksimum)
        peaks_thr = np.where(values > thr)
        peaks_thr2 = peaks[peaks_thr]

        '''plt.plot(np.arange(len(data_pt)), data_pt)
        plt.scatter(peaks_thr2, data_pt[peaks_thr2])
        plt.axhline(thr)
        plt.show()'''

        return peaks_thr2

    def T_point_detection(self):
        data = self.data_ecg
        R_points = self.R_peak_detection()
        RR_ints = []
        T_points = []
        for i in range(len(R_points) - 1):
            R_start = R_points[i]
            R_end = R_points[i + 1]
            RR_interval = R_end - R_start
            RR_ints.append(RR_interval)
            peaks, _ = find_peaks(data[(R_start + 1): (R_start + 1 + int(1 / 2 * RR_interval))])
            peaks = peaks + R_start + 1
            peak_T = np.argmax(data[peaks])
            # T_point = np.argmax(data[R_start + 1: (R_start + 1 + int(1 / 3 * RR_interval))]) + R_start + 1
            T_points.append(peaks[peak_T])

        # the last point
        RR_interval = np.mean(np.array(RR_ints))
        R_start = R_points[-1]
        peaks, _ = find_peaks(data[(R_start + 1): (R_start + 1 + int(1 / 3 * RR_interval))])
        peaks = peaks + R_start + 1
        peak_T = np.argmax(data[peaks])
        T_points.append(peaks[peak_T])
        return np.array(T_points)

    def T_end(self):
        T_points = self.T_point_detection()
        T_ends = []
        for i in range(len(T_points)):
            T_point = T_points[i]
            minimum = False
            j = 0
            while(minimum == False):
                fp = self.data_ecg[T_point + j]
                mp = self.data_ecg[T_point + j + 1]
                lp = self.data_ecg[T_point + j + 2]
                if (mp < fp and mp < lp and mp<0):
                    minimum = True
                    index = T_point + j + 1
                    T_ends.append(index)
                j += 1
        return np.array(T_ends)



    def S_point_detection(self):
        R_points = self.R_peak_detection()
        S_points = []
        for i in range(len(R_points)):
            R_point = R_points[i]
            minimum = False
            j = 0
            while(minimum==False):
                fp = self.data_ecg[R_point+j]
                mp = self.data_ecg[R_point+j+1]
                lp = self.data_ecg[R_point+j+2]
                if(mp < fp and mp < lp):
                    minimum = True
                    index = R_point + j+1
                j += 1
            S_points.append(index)
        return np.array(S_points)


    def C_point_detection(self):
        R_points = self.R_peak_detection()
        C_points = []
        cc_interval = []
        for i in range(len(R_points)-1):
            pos0 = R_points[i]
            posk = R_points[i+1]
            cc = self.data_icg[pos0:posk]
            cc_interval.append(posk-pos0)
            C_point = np.argmax(cc) + pos0
            C_points.append(C_point)
        c_mean = np.mean(np.array(cc_interval))
        pos0 = R_points[-1]
        posk = int(pos0 + c_mean)
        cc = self.data_icg[pos0:posk]
        C_point = np.argmax(cc) + pos0
        C_points.append(C_point)
        return np.array(C_points)

    def X_point_detection(self):
        T_points = self.T_end()
        X_points = []

        # first_derivative = np.gradient(self.data_icg)
        # second_derivative = np.gradient(first_derivative)

        for i in range(len(T_points)):
            T_point = T_points[i]
            minimum = False
            j = 0
            while(minimum == False):
                fp = self.data_icg[T_point+j]
                mp = self.data_icg[T_point+j+1]
                lp = self.data_icg[T_point+j+2]
                if(mp < fp and mp < lp):
                    minimum = True
                    index = T_point + j + 1
                j += 1
            X_points.append(index)
        return np.array(X_points)

    def B_point_detection(self):
        first_der = np.gradient(self.data_icg)
        second_der = np.gradient(first_der)
        S_points = self.S_point_detection()
        B_points = []
        for i in range(len(S_points)):
            S_point = S_points[i]
            maks = False
            j = 0
            k = 0
            while(maks == False):
                fp = second_der[S_point + j]
                mp = second_der[S_point + j + 1]
                lp = second_der[S_point + j + 2]
                if(mp > fp and mp> lp and k == 0):
                    k += 1
                elif(mp > fp and mp > lp and k == 1):
                    maks = True
                    index = S_point + j + 1
                    B_points.append(index)
                j += 1
        return np.array(B_points)



