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
from apc_model import APCModel
from utils import PrenetConfig, RNNConfig

from tqdm import tqdm

import numpy as np
class IEMOCAP(data.Dataset):

    # root： 实验数据的文件夹名; mode: train val test
    def __init__(self, experiment_name, mode='train', transform = None):
        # 相对转绝对T
        self.mfcc_datafolder = '../dataset/'
        self.emotion_classes = {"ang": 0, "hap": 1,
                                "neu": 2, "sad": 3}
        self.fd_root = '../feature_data/' + experiment_name
        self.mode = mode
        if not os.path.exists(self.fd_root+'_' + self.mode):
            self.compute_mspc()
            self.compute_mfcc()
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

        for emotion_name in tqdm(os.listdir(self.mfcc_datafolder)):
            temp_list = []
            if emotion_name.startswith('.'):
                continue
            emotion_folder = '../dataset/' + emotion_name + '/'
            for wav_file in tqdm(os.listdir(emotion_folder)):
                if wav_file.startswith('.'):
                    continue
                wav_path = emotion_folder + wav_file
                wav_sr = 16000
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

    # 随机划分数据集
    def compute_mspc(self):
        emotion_data = {}
        random.seed(123456)
        prenet_config = None	
        rnn_config = RNNConfig(input_size=80, hidden_size=512, num_layers=3, residual=True, dropout=0.)
        pretrained_apc = APCModel(mel_dim=80, prenet_config=prenet_config, rnn_config=rnn_config).cuda()
        pretrained_weights_path = 'bs32-rhl3-rhs512-rd0-adam-res-ts20.pt'
        pretrained_apc.load_state_dict(torch.load(pretrained_weights_path))
        device=torch.device(0) 
        example_path = '/home/hongyuan/Desktop/mlp_groupcw/framework/result/result_mockingjay/mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt'
        mockingjay = get_mockingjay_model(from_path=example_path)

        for emotion_name in tqdm(os.listdir(self.mfcc_datafolder)):
            temp_list = []
            if emotion_name.startswith('.'):
                continue
            emotion_folder = '../dataset/' + emotion_name + '/'
            for wav_file in tqdm(os.listdir(emotion_folder)):
                if wav_file.startswith('.'):
                    continue
                wav_path = emotion_folder + wav_file
                wav_sr = 16000
                y, sr = librosa.load(path=wav_path, sr=wav_sr)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40))
                #print(mfccs.shape)
                mel=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=160, n_fft=int(sr/40), hop_length=int(sr/100))
                mel=np.expand_dims(mel, axis=0)
                mel=np.transpose(mel, (0, 2, 1))
                
                mel=torch.Tensor(mel)
                print([mel.shape[1]])
                mel=mel.to('cpu')
                length=[mel.shape[1]]
                length=torch.Tensor(length)
                length=length.to('cpu')

                #_, mel = pretrained_apc.forward(mel, length)
                                # reps.shape: (batch_size, seq_len, hidden_size)
                mel = mockingjay.forward(spec=mel, all_layers=False, tile=True)
                #mel=mel[-1,-1,:,:]
                mel=mel[-1,:,:]
                
                mel=mel.transpose(1, 0)
                mel=mel.to("cpu")
                mel=mel.detach() 
   
                print(mel.shape)
                labels = self.emotion_classes[emotion_name]
     
                temp_list.append((mel, labels))
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

    ama = IEMOCAP('mpc',mode='train')
