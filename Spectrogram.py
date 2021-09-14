import math
import os.path
import matplotlib.pyplot as plt
import numpy as np
import librosa
from pathlib import Path
from scipy import signal
import scipy.fft

# function spectorgram
# creates a spectrogram
# loads audio file, 'divides' into windows, calculates and plots spectrographic features
#     window_size -> number of samples to account for in each spectrogram window
#     window_overlap_percentage -> percentage of window_size that overlaps between windows (i.e, step size)
#     min_frequency -> minimum frequency cut-off value for the spectrogram
#     max_frequency -> maximum frequency cut-off value for the spectrogram
#     threshold -> spectrographic values (i.e., volume) cut-off value for the spectrogram
def spectrogram(
        audio_file_path, label_file_path, window_size=256, window_overlap_percentage=50,
        min_frequency=10, max_frequency=8000, threshold=0.1):
    samples, sample_rate, duration = load_audio(audio_file_path)
    windows, stride_size = get_strided_windows(samples, window_size, window_overlap_percentage)
    spectrographic_features = calculate_spectrographic_features(
        windows, window_size, sample_rate,
        min_frequency, max_frequency, threshold)
    create_labels(len(np.swapaxes(spectrographic_features, 0, 1)), duration, audio_file_path, label_file_path)
    plot_spectrogram(spectrographic_features, audio_file_path)


# Plots a spectrogram using matplotlib
def plot_spectrogram(spectrographic_features, audio_file_path):
    # print("Spectrographic Features: {}"
    #      .format(np.swapaxes(spectrographic_features, 0, 1)[0]))
    np.savetxt("{}_spectrographic_features_flipped".format(audio_file_path).replace(".wav", "")+".csv", spectrographic_features, delimiter=",")
    plt.figure(figsize=(50, 3))
    plt.imshow(spectrographic_features, aspect='auto')
    plt.colorbar()
    plt.set_cmap('magma')
    plt.show()


def create_labels(window_count, duration, audio_file_path, label_file_path):
    labels = np.zeros([window_count])
    with open(label_file_path) as f:
        file = f.read()
        print(file)
    for line in file.split('\n'):
        values = line.split('\t')
        start = math.ceil(float(values[0]) / duration * window_count)
        print(start)
        end = math.ceil(float(values[1]) / duration * window_count)
        print(end)
        label = get_label_index(values[2])
        for i in range(start, end):
            labels[i] = label
    np.savetxt("{}_labels".format(audio_file_path).replace(".wav", "") + ".csv", labels, delimiter=",")


def get_label_index(label_name):
    if label_name == "Object":
        return 1
    elif label_name == "Human":
        return 2
    elif label_name == "Chicken":
        return 3
    elif label_name == "Click":
        return 4
    else:
        return 0


# creates array of windows using an array of samples: cuts off the modulo from the array of samples,
# then creates windows of size window_size with step stride_size
def get_strided_windows(
        samples, window_size, window_overlap_percentage):
    stride_size = int(window_overlap_percentage / 100 * window_size)
    cut_off = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - cut_off]
    shape = (window_size, (len(samples) - window_size) // stride_size + 1)
    strides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    print("Window Count: {}"
          .format(len(windows[0])))
    return windows, stride_size


# calculates the spectrographic values needed for the spectrogram: creates a gaussian filter that is used for the
# weighting of the windows, calculates the Fast Fourier Transform of each window to find the frequency distribution,
# crops the frequencies, scales magnitudes to dB and performs a series of normalization and correction calculations
def calculate_spectrographic_features(
        windows, window_size, sample_rate,
        min_frequency, max_frequency, threshold):
    gaussian_filter = signal.windows.gaussian(M=window_size, std=max_frequency)[:, None]
    fft = scipy.fft.rfft(windows * gaussian_filter, axis=0)
    fft **= 2
    scale = np.sum(gaussian_filter ** 2) * sample_rate
    fft[1:-1] *= (2.0 / scale)
    fft[(0, -1)] /= scale
    fft = crop_frequencies(fft, float(sample_rate) / window_size, min_frequency, max_frequency)
    spectrographic_features = magnitude_to_db(fft)
    spectrographic_features_modified = correct_spectrographic_features(spectrographic_features, threshold)
    return spectrographic_features_modified


# corrects spectrographic values: normalizes the values using a combination of different methods, shifts the mean of
# the spectrogram, cuts off values below threshold and flips the spectrogram so it has the correct orientation
def correct_spectrographic_features(spectrographic_features, threshold):
    s_f_modification_1 = normalize_per_column(spectrographic_features) / get_mean_per_column(spectrographic_features)
    s_f_modification_2 = normalize_complete_array(spectrographic_features)
    spectrographic_features_modified = normalize_complete_array(s_f_modification_1 + s_f_modification_2)
    spectrographic_features_modified = shift_spectrogram_values(spectrographic_features_modified)
    spectrographic_features_modified[np.where(spectrographic_features_modified < threshold)] = threshold
    spectrographic_features_modified = normalize_complete_array(spectrographic_features_modified ** (1/2))
    spectrographic_features_modified = np.flipud(spectrographic_features_modified)
    return spectrographic_features_modified


# shifts spectrogram values so the mean value is 0.5, then clips it to 0,1 range (warning: cuts off values)
def shift_spectrogram_values(spectrographic_features_modified):
    spectrogram_value_shift = (get_mean_complete_array(spectrographic_features_modified) - 0.5) * -2
    spectrographic_features_modified \
        = (1 - spectrogram_value_shift) * spectrographic_features_modified + spectrogram_value_shift
    spectrographic_features_modified = np.clip(spectrographic_features_modified, 0, 1)
    return spectrographic_features_modified


# crops frequencies to desired range
def crop_frequencies(fft, window_count, bottom_cut_off, top_cut_off):
    frequency_array = window_count * np.arange(fft.shape[0])
    top_cut_off_index = np.where(frequency_array <= top_cut_off)[0][-1] + 1
    bottom_cut_off_index = np.where(frequency_array >= bottom_cut_off)[0][0]
    return fft[bottom_cut_off_index:top_cut_off_index]


# converts magnitude values to dB values
def magnitude_to_db(magnitudes):
    return np.abs(10 * np.log10(magnitudes))


# returns the median value of each column
def get_mean_per_column(array):
    return np.mean(array, axis=0)


# returns the median value of each column
def get_mean_complete_array(array):
    return np.mean(array)


# normalizes values in columns independently from each other to 0-1
def normalize_per_column(array):
    array_normalized = array - np.min(array, axis=0)
    array_normalized /= np.max(array, axis=0) - np.min(array)
    return array_normalized


# normalizes values of entire array to 0-1
def normalize_complete_array(array):
    array_normalized = array - np.min(array)
    array_normalized /= np.max(array) - np.min(array)
    return array_normalized


# loads audio data and prints its information
def load_audio(path):
    samples, sample_rate = librosa.load(path)
    duration = len(samples) / sample_rate
    print("Loaded audio file \"{}\""
          "\n\tFull path: \"{}\""
          "\n\tSample rate: {}"
          "\n\tDuration: {:.1f}s"
          .format(Path(path).stem,
                  os.path.abspath(path),
                  sample_rate,
                  duration))
    return samples, sample_rate, duration


audio_file_path = "Audio/ch07-20210224-093335-094153-001000000000.wav"
label_file_path = "Audio/l_ch07-20210224-093335-094153-001000000000.txt"
spectrogram(audio_file_path, label_file_path)
