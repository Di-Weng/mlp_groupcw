# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import pickle
import librosa
import random
import os
from torch.utils import data
import torch

import numpy as np
class IEMOCAP(data.Dataset):

    # root： 实验数据的文件夹名; mode: train val test
    def __init__(self, experiment_name, mode='train', transform = None):
        # 相对转绝对T
        self.mfcc_datafolder = '../dataset/'
        self.emotion_classes = {"ang": torch.Tensor([1, 0, 0, 0]), "hap": torch.Tensor([0, 1, 0, 0]),
                                "neu": torch.Tensor([0, 0, 1, 0]), "sad": torch.Tensor([0, 0, 0, 1])}
        self.fd_root = '../feature_data/' + experiment_name
        self.mode = mode
        if not os.path.exists(self.fd_root+'_' + self.mode):
            self.compute_mfcc()
        else:
            with open(self.fd_root+'_' + self.mode,'rb') as f:
                self.data = pickle.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx][0], self.data[idx][1]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    # 随机划分数据集
    def compute_mfcc(self):
        emotion_data = {}
        random.seed(123456)

        for emotion_name in os.listdir(self.mfcc_datafolder):
            temp_list = []
            if emotion_name.startswith('.'):
                continue
            emotion_folder = '../dataset/' + emotion_name + '/'
            for wav_file in os.listdir(emotion_folder):
                if wav_file.startswith('.'):
                    continue
                wav_path = emotion_folder + wav_file
                wav_sr = librosa.get_samplerate(wav_path)
                y, sr = librosa.load(path=wav_path, sr=wav_sr)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40))
                labels = self.emotion_classes[emotion_name]
                temp_list.append((torch.Tensor(mfccs), labels))
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
        return self.data
if __name__ == '__main__':

    ama = IEMOCAP('mfcc',mode='test')

