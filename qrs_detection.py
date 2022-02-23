import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class qrs ():

    def __init__(self, data_ecg, fs):
        self.data_ecg = data_ecg
        self.fs = fs

    def enhancement_mask(self):
        new_ecg = np.zeros(len(self.data_ecg))
        for i in range(1, len(self.data_ecg)-1):
            fp = self.data_ecg[i-1]*(-1)
            sp = self.data_ecg[i]*2
            tp = self.data_ecg[i+1]*(-1)
            new_ecg[i] = fp + sp + tp
        new_ecg = 2*(new_ecg-np.min(new_ecg))/(np.max(new_ecg)-np.min(new_ecg)) - 1
        return new_ecg

    def _plot_creast_trough(self, segment, peaks_crest, peaks_troughs):
        plt.plot(np.arange(len(segment)), segment)
        plt.scatter(peaks_crest, segment[peaks_crest])
        plt.scatter(peaks_troughs, segment[peaks_troughs])
        plt.show()

    def crest_and_troughs(self, plot=False):
        searching_range = int(0.3*self.fs)
        data = self.enhancement_mask()
        print(len(data))
        max_thr = 0.22
        min_thr = 0.4
        thr2 = 0.52
        k = 0
        qrs_points = []
        lista = list(data > max_thr)
        start_point = lista.index(next(filter(lambda i: i == 1, lista)))
        while(start_point + searching_range < len(data)):
            # print(k)
            segment = data[start_point: (start_point+searching_range)]
            peaks = find_peaks(segment)[0]
            troughs = find_peaks(-segment)[0]
            peaks_crest = peaks[segment[peaks]>0.22]
            peaks_troughs = troughs[segment[troughs]<-0.4]
            # cases:
            n_creasts = len(peaks_crest)
            n_troughs = len(peaks_troughs)
            if(n_creasts == 1 and n_troughs == 1):
                QRS = peaks_crest[0] + start_point
                qrs_points.append(QRS)
                start_point = peaks_troughs[0] + start_point
                if(plot):
                    self._plot_creast_trough(segment, peaks_crest, peaks_troughs)
                k += 1
            elif(n_creasts == 2 and n_troughs == 1 and peaks_troughs[0]>peaks_crest[0] and peaks_troughs[0]<peaks_crest[1]):
                QRS = peaks_troughs[0] + start_point
                qrs_points.append(QRS)
                start_point = peaks_troughs[0] + int(0.12 * self.fs)
                if(plot):
                    self._plot_creast_trough(segment, peaks_crest, peaks_troughs)
                k += 1
            elif(n_creasts == 1 and segment[peaks_crest[0]] > thr2):
                QRS = peaks_crest[0] + start_point
                qrs_points.append(QRS)
                if(plot):
                    self._plot_creast_trough(segment, peaks_crest, peaks_troughs)
                start_point = peaks_crest[0] + int(0.12 * self.fs)
                k += 1
            else:
                if(plot):
                    self._plot_creast_trough(segment, peaks_crest, peaks_troughs)
                lista = list(data[(start_point+1):] > max_thr)
                try:
                    start_point = lista.index(next(filter(lambda i: i == 1, lista))) + start_point + 1
                except StopIteration:
                    break
                k += 1
        return np.array(qrs_points)

    def S_point_detection(self):
        fiducial_points = self.crest_and_troughs()
        data = self.enhancement_mask()
        offset = int(0.12 * self.fs)
        S_points = []

        for i in range(len(fiducial_points)):
            fp = int(fiducial_points[i])
            S_found = False
            while(S_found == False):
                segment_statistics = data[(fp-offset): (fp+offset)]
                '''step 1'''
                amplitude = np.max(segment_statistics)
                mean_amplitude = np.mean(segment_statistics)
                '''step 2'''
                S_start = find_peaks(data[fp: (fp+offset)], height=amplitude*0.5)[0]
                if(len(S_start)==0):
                    peaks_crest = data[fp:(fp+offset)]
                    minimum = np.min(peaks_crest)
                    if(minimum < mean_amplitude):
                        S_points.append(np.argmin(peaks_crest) + fp)
                        S_found = True
                else:
                    fp = S_start

        return np.array(S_points) + 2

    def S_offset(self):
        data = self.enhancement_mask()
        S_points = self.S_point_detection()
        first_der = np.gradient(data)
        S_offsets = []
        for i in range(len(S_points)):
            sp = S_points[i]
            next_p = sp + 1
            val_sp = first_der[sp]
            val_np = first_der[next_p]
            while(val_sp *  val_np>0):
                sp = next_p
                next_p = sp + 1
                val_sp = first_der[sp]
                val_np = first_der[next_p]
            S_offsets.append(next_p)
        return np.array(S_offsets)









