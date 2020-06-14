import tensorflow as tf

import os
import pickle
from keras.callbacks import ModelCheckpoint
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, TimeDistributed, UpSampling2D
from keras.models import Sequential
from keras import backend as K


from keras.applications import ResNet50
from keras_radam import RAdam

from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from cfg import config
import librosa
import scipy

import math

import functools
import operator



#%%


TESTING_CSV_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.meta/train_post_competition.csv'
CLEANED_FILES_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/DATASETS/FSD/FSDKaggle2018.audio_train/'

# TESTING_CSV_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/train_mini_dataset.csv'
# CLEANED_FILES_PATH = 'D:/Sklad/Jan 19/RTU works/3_k_sem_1/Bakalaura Darbs/-=Python Code=-/-=2020=-/graduation_project/data_stuff/mini_dataset/wavfiles/'

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
        # with open(config.fast_p_path, 'rb') as handle:
        #     tmp = pickle.load(handle)
        #     return tmp
        
        
    else:
        return None
        
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    
    
    X = []
    y = []  
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read(CLEANED_FILES_PATH + file)
        
        # sample = librosa.resample(librosa.util.buf_to_float(wav), rate, 16000)
        # split_points = librosa.effects.split(sample, top_db=80, frame_length=1024, hop_length=512)
           
        # S_cleaned = []
         
        # for piece in split_points:
  
        #     S_cleaned.append(sample[piece[0]:piece[1]])
        
        # sample = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
        
        
        
        
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        
       
        
        #float_audio = librosa.util.buf_to_float(sample)
        
        #sample = librosa.util.normalize(sample)
        
        # sampleRate = rate 
        # cutOffFrequency = 8000
        # freqRatio = (cutOffFrequency/sampleRate)

        # N = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)

        # win = np.ones(N)
        # win *= 1.0/N
        # sample = scipy.signal.lfilter(win, [1], sample)
        # sample = librosa.resample(sample, rate, 16000)
        
        # split_points = librosa.effects.split(sample, top_db=80, frame_length=1024, hop_length=512)
           
        # S_cleaned = []
         
        # for piece in split_points:
  
        #     S_cleaned.append(sample[piece[0]:piece[1]])
        
        # sample = np.array(functools.reduce(operator.iconcat, S_cleaned, []))
        
        
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft = config.nfft)
        
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    
    config.min = _min
    config.max = _max
        
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode =='resnet':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    y = to_categorical(y, num_classes=41)
    
    config.data = (X,y)
    
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    
    return X, y

resnet_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 1))    

def get_resnet_model():
    model = Sequential()
    
    #model.add(UpSampling2D())
    model.add(UpSampling2D())
    model.add(resnet_model)
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(41, activation='softmax'))
    #model.summary()
    model.compile(RAdam(),loss='categorical_crossentropy',
                  #optimizer='adam',
                  metrics=['acc',f1_m,precision_m, recall_m])
    return model

def get_conv_model():
    model = Sequential()
    
    
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1),
                      padding='same', input_shape=input_shape))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                      padding='same'))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                      padding='same')) 
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                      padding='same'))
    
    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    model.summary()
    model.compile(RAdam(),loss='categorical_crossentropy',
                  #optimizer='adam',
                  metrics=['acc',f1_m,precision_m, recall_m])
    return model






# class Config:
#     def __init__(self, mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
#         self.mode  = mode
#         self.nfilt = nfilt
#         self.nfeat = nfeat
#         self.nfft = nfft
#         self.rate = rate
#         self.step = int(rate/10)

df = pd.read_csv(TESTING_CSV_PATH)
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read(CLEANED_FILES_PATH + f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1) #why do we have coefficent 2 here? *Make sure we have dataset big enough



prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()



config = config(mode='resnet')

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model() # to be added later
    
elif config.mode == 'resnet':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis = 1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_resnet_model() # to be added later
    
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=3)

model.fit(X, y, epochs=14, batch_size=256,shuffle=True,
          validation_split=0.3, callbacks=[checkpoint])

model.save(config.model_path)    
## asdf


