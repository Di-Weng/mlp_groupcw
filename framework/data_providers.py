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
import numpy as np
class IEMOCAP(data.Dataset):

    # root： 实验数据的文件夹名
    def __init__(self, experiment_name):
        # 相对转绝对
        self.mfcc_datafolder = '../dataset/'
        self.emotion_classes = {"ang": np.array([1, 0, 0, 0]), "hap": np.array([0, 1, 0, 0]),
                                "neu": np.array([0, 0, 1, 0]), "sad": np.array([0, 0, 0, 1])}

        self.fd_root = '../feature_data/' + experiment_name
        if not os.path.exists(self.fd_root):
            self.compute_mfcc()
        with open(self.fd_root, 'rb') as f:
            self.emotion_data = pickle.load(f)
        self.train_data = []
        self.val_data = []
        self.test_data = []
        random.seed(123456)
        for emotion, emotion_data_list in self.emotion_data.items():
            random.shuffle(emotion_data_list)
            length = len(emotion_data_list)
            self.train_data.extend(emotion_data_list[:int(length * 0.7)])
            self.val_data.extend(emotion_data_list[int(length * 0.7):int(length * 0.9)])
            self.test_data.extend(emotion_data_list[int(length * 0.9):])

        random.shuffle(self.train_data)
        random.shuffle(self.val_data)
        random.shuffle(self.test_data)

        self.train_data = np.asarray(self.train_data)
        self.val_data = np.asarray(self.val_data)
        self.test_data = np.asarray(self.test_data)

    def compute_mfcc(self):
        data = {}
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
                temp_list.append((np.array(mfccs, dtype=np.float32), labels))
            data[emotion_name] = temp_list
        with open(self.fd_root, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        f.close()

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(5531)
        fmt_str += '    Data Location: {}\n'.format(self.fd_root)
        return fmt_str

    def test_1(self):
        return self.train_data[3]
    def test(self):
        with open(self.fd_root, 'rb') as f:
            self.emotion_data = pickle.load(f)
        for idx, (x, y) in enumerate(self.train_data):
            print(idx)
            print(type(x))
            print(type(y))
if __name__ == '__main__':
    ama = IEMOCAP('mfcc')
    train_1 = ama.test_1()
    ama.test()


