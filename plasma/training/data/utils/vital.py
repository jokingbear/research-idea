import numpy as np
import scipy.interpolate as intp

from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt


def butter_bandpass(sig, fs, lowcut, highcut, order=1):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y


def hr_fft(sig, fs, lowcut=0.6, highcut=4, harmonics_removal=False, return_psd=False):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(lowcut / fs * sig.shape[0]).astype('int')
    high_idx = np.round(highcut / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1 - 2 * hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    if not return_psd:
        return hr

    x_hr = np.arange(len(sig)) / len(sig) * fs * 60
    return hr, sig_f_original, x_hr


def interpolate(signal, original_fs, new_fs):
    time = np.arange(len(signal)) / original_fs
    duration = len(signal) / original_fs

    func = intp.interp1d(time, signal)

    new_time = np.arange(0, duration, step=1 / new_fs)

    return func(new_time)
