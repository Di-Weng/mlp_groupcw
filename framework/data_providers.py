# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""
import torch
from runner_mockingjay import get_mockingjay_model
import pickle
import librosa
import random
import os
from torch.utils import data
import torch

from tqdm import tqdm

import numpy as np

sample_rate = 16000
"""
For feature == 'fbank' or 'mfcc'
"""
num_mels = 80 # int, dimension of feature
delta = True # Append Delta
delta_delta = False # Append Delta Delta
window_size = 25 # int, window size for FFT (ms)
stride = 10 # int, window stride for FFT
mel_dim = num_mels * (1 + int(delta) + int(delta_delta))
"""
For feature == 'linear'
"""
num_freq = 1025
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
hop_length = 250
griffin_lim_iters = 16
power = 1.5 # Power to raise magnitudes to prior to Griffin-Lim
"""
For feature == 'fmllr'
"""
fmllr_dim = 40

def _stft_parameters(sample_rate):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length

def _linear_to_mel(spectrogram, sample_rate):
    _mel_basis = _build_mel_basis(sample_rate)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(sample_rate):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

def _preemphasis(x):
    return signal.lfilter([1, -preemphasis], [1], x)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _stft(y, sr):
    n_fft, hop_length, win_length = _stft_parameters(sr)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


#############################
# SPECTROGRAM UTILS BACKWARD #
#############################
def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def inv_preemphasis(x):
    return signal.lfilter([1], [1, -preemphasis], x)

def _griffin_lim(S, sr):
    """
        librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, sr)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y ,sr)))
        y = _istft(S_complex * angles, sr)
    return y

def _istft(y, sr):
    _, hop_length, win_length = _stft_parameters(sr)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


###################
# MEL SPECTROGRAM #
###################
"""
Compute the mel-scale spectrogram from the wav.
"""
def melspectrogram(y, sr):
    D = _stft(_preemphasis(y), sr)
    S = _amp_to_db(_linear_to_mel(np.abs(D), sr))
    return _normalize(S)


###############
# SPECTROGRAM #
###############
"""
Compute the linear-scale spectrogram from the wav.
"""
def spectrogram(y, sr):
    D = _stft(_preemphasis(y), sr)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


###################
# INV SPECTROGRAM #
###################
"""
Converts spectrogram to waveform using librosa
"""
def inv_spectrogram(spectrogram, sr=16000):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** power, sr))          # Reconstruct phase


def extract_feature(input_file, feature='fbank', cmvn=True, save_feature=None):
    y, sr = librosa.load(input_file, sr=sample_rate)

    if feature == 'fbank':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6) # log-scaled
    elif feature == 'mfcc':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mels, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rms(y, hop_length=st, frame_length=ws)
    elif feature == 'mel':
        # feat = melspectrogram(y, sr) # deprecated
        n_fft, hop_length, win_length = _stft_parameters(sr)
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels,
                                              n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        feat = np.log(feat + 1e-6) # log-scaled
    elif feature == 'linear':
        feat = spectrogram(y, sr)
    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    # Apply delta
    feat = [feat]
    if delta and feature != 'linear':
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta and feature != 'linear':
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if feature == 'linear': assert(np.shape(feat)[0] == num_freq)

    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')

class IEMOCAP(data.Dataset):


    # root： 实验数据的文件夹名; mode: train val test
    def __init__(self, experiment_name, layer_no, mode='train', transform = None):
        # 相对转绝对T
        self.mfcc_datafolder = '../dataset/'
        self.emotion_classes = {"ang": 0, "hap": 1,
                                "neu": 2, "sad": 3}
        self.fd_root = '../feature_data/' + experiment_name
        if experiment_name=="mpc":
            self.fd_root = '../feature_data/' + experiment_name + str(layer_no)
        self.mode = mode
        if not os.path.exists(self.fd_root+'_' + self.mode):
            self.compute_feature(experiment_name, layer_no)
        with open(self.fd_root+'_' + self.mode,'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, z = self.data[idx][0], self.data[idx][1], self.data[idx][2]

        if self.transform is not None:
            x = self.transform(x)

        return x, y, z

    def compute_feature(self, experiment_name, layer_no):
        emotion_data = {}
        random.seed(123456)
        prenet_config = None

        if experiment_name=="mpc" or experiment_name=="mpc_all_layers" :
            model_path = 'MPC/mockingjay-500000.ckpt'
            mockingjay = get_mockingjay_model(from_path=model_path)
        for emotion_name in tqdm(os.listdir(self.mfcc_datafolder)):
            temp_list = []
            if emotion_name.startswith('.'):
                continue
            emotion_folder = '../dataset/' + emotion_name + '/'
            for wav_file in tqdm(os.listdir(emotion_folder)):
                if wav_file.startswith('.'):
                    continue
                gender=wav_file[-8]
                #print(gender)
                if gender == 'F':
                    gender = 0
                elif gender == 'M':
                    gender = 1
                assert gender == 0 or gender == 1
                #print(gender)
                wav_path = emotion_folder + wav_file
                wav_sr = 16000
                y, sr = librosa.load(path=wav_path, sr=wav_sr)
                feature=None
                #print(experiment_name)
                if experiment_name=="mfcc":
                    feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40))
                    #print(feature.shape)
                if experiment_name=="mspc":
                    feature=extract_feature(wav_path, feature='mel', cmvn=True, save_feature=None)
                    feature=feature.transpose(1, 0)
                    #print(feature.shape)
                if experiment_name=="mpc":
                    feature = extract_feature(wav_path, feature='mel', cmvn=True, save_feature=None)
                    feature=np.expand_dims(feature, axis=0)
                    #print(feature.shape)

                    #feature=np.transpose(feature, (0, 2, 1))
                    #print(feature.shape)
                    feature=torch.Tensor(feature)
                    feature=feature.to('cpu')
                    
                    length=[feature.shape[1]]
                    length=torch.Tensor(length)
                    length=length.to('cpu')


                #_, mel = pretrained_apc.forward(mel, length)
                                # reps.shape: (batch_size, seq_len, hidden_size)
                    feature = mockingjay.forward(spec=feature, all_layers=True, tile=True)
                    #mel=mel[-1,-1,:,:]
                    feature=feature[-1,layer_no,:,:]
                
                    feature=feature.transpose(1, 0)
                    #print(feature.shape)
                    feature=feature.to("cpu")
                    feature=feature.detach()
                    #print(feature.shape)
                #print(emotion_name)
   
                emotion = self.emotion_classes[emotion_name]
     
                temp_list.append((feature, emotion, gender))
                #print("actually: "+ str(temp_list[0][2]))
            emotion_data[emotion_name] = temp_list

        train_data = []
        val_data = []
        test_data = []

        for emotion, emotion_data_list in emotion_data.items():
            random.shuffle(emotion_data_list)
            length = len(emotion_data_list)
            train_data.extend(emotion_data_list[:int(length * 0.7)])
            val_data.extend(emotion_data_list[int(length * 0.7):int(length * 0.9)])
            test_data.extend(emotion_data_list[int(length * 0.9):])

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        with open(self.fd_root+'_train', 'wb') as f:
            pickle.dump(train_data, f, protocol=4)
        with open(self.fd_root + '_val', 'wb') as f:
            pickle.dump(val_data, f, protocol=4)
        with open(self.fd_root + '_test', 'wb') as f:
            pickle.dump(test_data, f, protocol=4)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(5531)
        fmt_str += '    Data Location: {}\n'.format(self.fd_root)
        return fmt_str

    def test(self):

        return [temp[0] for temp in self.data]

if __name__ == '__main__':
    for i in np.arange(12):
        ama = IEMOCAP(experiment_name='mpc',layer_no=11, mode='train')
