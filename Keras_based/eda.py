import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

#%%

## plotting methods


TRAIN_CSV_PATH = "D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/train_mini_dataset.csv"
TRAIN_FILES_PATH = "D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/wavfiles/"
TRAIN_FILES_CLEANED = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/20_classes_train/'

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
            

## The function, getting envelope of a signal by given threshold

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)           
        else:
            mask.append(False)            
    return mask
    


## the function to caculate fft for certain signal (as array) at known samplerate, returns magnitudes vs frequencies
            
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)
    
##reading data .csv file, settin index

df = pd.read_csv(TRAIN_CSV_PATH)
df.set_index('fname',inplace =True)

#firstindex = df.index.values[0]
#rate, signal = wavfile.read('wavfiles/' + firstindex)
#check_me = signal.shape[0]
#check_me2 = signal.shape
#df.at[firstindex,'length'] = signal.shape[0]/rate

## adding new column, containing file lengths in seconds

for f in df.index:
    rate, signal = wavfile.read(TRAIN_FILES_PATH + f)
    df.at[f,'length'] = signal.shape[0]/rate
    
#test_label_array = df.label
    
## getting average file lenght in every category, creating class_dist dataframe  
    
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

## pie chart plotting, showing category distribution

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels=class_dist.index,autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True)

## dictionaries

signals = {}
fft = {}
fbank = {}
mfccs = {}

## show the first file of every class as signal, fft, spectrogram and mfccs

#%%

for c in classes:
    wav_file = df[df.label == c].iloc[0,0]
    signal, rate = librosa.load(TRAIN_FILES_PATH + wav_file, sr=44100)
    
    mask = envelope(signal, rate, 0.0005)
    
    signal = signal[mask]
    
    signals[c] = signal
    fft[c] = calc_fft(signal,rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

    
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

#%%
#
## use envelope function and write "cleaned" files into directory


# if len(os.listdir(TRAIN_FILES_CLEANED)) == 0:
#     for f in tqdm(df.fname):
#         signal, rate = librosa.load(TRAIN_FILES_PATH + f, sr=16000)
#         mask = envelope(signal, rate, 0.0005)
#         wavfile.write(filename=TRAIN_FILES_CLEANED + f, rate=rate, data=signal[mask])
        
#%% copy only required categories

SOURCE_TEST_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/'         
CLASSES20_TEST_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/20_classes_train/'
CLASSES20_CSV_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_20_classes.csv'

df = pd.read_csv(CLASSES20_CSV_PATH)
df.set_index('fname',inplace =True)

#%%



# if len(os.listdir(CLASSES20_TEST_PATH)) == 0:
#     for f in tqdm(df.index):
#         signal, rate = librosa.load(SOURCE_TEST_PATH + f, sr=16000)
#         mask = envelope(signal, rate, 0.00005)
#         wavfile.write(filename=CLASSES20_TEST_PATH + f, rate=rate, data=signal[mask])

#%%

# TRAIN_CSV_PATH = "D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_20_classes.csv"
# path = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/'

# df = pd.read_csv(TRAIN_CSV_PATH)
# df.set_index('fname',inplace =True)
# classes = list(np.unique(df.label))
# class_dist = df.groupby(['label']).agg(['count'])['freesound_id']
# values = class_dist['count']

# #%%
# fig = plt.figure(figsize=[9, 9])
# ax = fig.add_subplot(111)

# ax.pie(values, labels=class_dist.index, autopct=lambda p:f'{p*sum(values)/100 :.0f} ')

# #%%

# durations = []
# filenamez = []

# for f in df.index:
  
#         y, sr = librosa.load(path + f, sr=44100, mono=True)
#         n_fft = 1024
#         #S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft//2)
#         f_len = len(y)/sr
#         durations.append(f_len)
#         filenamez.append(f)
#         # convert to db
#         # (for your CNN you might want to skip this and rather ensure zero mean and unit variance)

# dur = pd.DataFrame({'duration': durations},  index = filenamez)    
