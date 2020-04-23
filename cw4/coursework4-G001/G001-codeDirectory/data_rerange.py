# -*- coding: utf-8 -*-
"""
-------------------------------
   Time    : 2020/2/4 1:08
   Author  : diw
   Email   : di.W@hotmail.com
   File    : data_rerange.py
   Desc:    re-range datafile according to its emotion class
            emotion_code={"ang":0 , "hap":1 , "neu":2 , "sad":3 }, exc is counted is hap
            xxx is disposed when using majority vote
-------------------------------
"""
import os
from shutil import copyfile
import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt

emotion_list=["ang", "hap", "sad",'neu']

duration_list = []

def load_audio():
    '''
    is_distribution = 0, return the most possible emotion class; else return the annotated distribution

    return: rerange the file as /session/emotion
    '''

    wav_folder = 'IEMOCAP/wav'
    EmoEval_folder = 'IEMOCAP/EmoEvaluation'

    #create class folder
    for emo in emotion_list:
        if not os.path.exists('dataset/' + emo):
            os.makedirs('dataset/' + emo)

    wav_folder_list = [foldername for foldername in os.listdir(wav_folder) if not foldername.startswith('.')] # remove index file which starts with .

    for wav_cut in wav_folder_list:

        # true file folder
        wav_file_folder = wav_folder + '/' + wav_cut

        with open(EmoEval_folder + '/'+ wav_cut + '.txt') as Evofile:
            for line in Evofile:
                line = line.strip()

                # filter out non-summary line
                if(not line.startswith('[')):
                    continue

                # apply to summary line below
                line_seg = line.split('\t')

                current_filename = line_seg[1]
                current_emo = line_seg[2]

                # transfer excitment into happy
                if(current_emo == 'exc'):
                    current_emo = 'hap'

                # filter out other emotions(only care about ["ang", "hap", "sad",'neu'])
                if(current_emo not in emotion_list):
                    continue

                src = wav_file_folder + '/' +  current_filename + '.wav'
                tgr = 'dataset/' + current_emo + '/' + current_filename + '.wav'

                copyfile(src, tgr)



def wav_duration():
    '''
    return: list , each item is the duration of the single wav file.
    '''
    time_list = []

    emo_list = os.listdir('dataset/')

    for emo in emo_list:

        emo_wav_list = os.listdir('dataset/' + emo)
        for emo_wav_file in emo_wav_list:

            # use wav
            # wav_file =  wave.open('dataset/' + emo + '/' + emo_wav_file,'rb')
            # wav_framerate = wav_file.getframerate()
            # wav_frame = wav_file.getnframes()
            # time = wav_frame / wav_framerate

            # use librosa
            time = librosa.get_duration(filename='dataset/' + emo + '/' + emo_wav_file)
            time_list.append(time)

    return time_list



if __name__=='__main__':

    # load_audio()
    time_list = wav_duration()
    time_list = np.asarray(time_list)
    print(time_list)
    plt.hist(time_list, bins=100)
    plt.xlabel('wav duration')
    plt.ylabel('count')
    plt.savefig('img/dura_distribution.png')
    plt.title('IEMOCAP wav file duration distribution')

    with open('result/dura_distri.txt','w') as f1:
        f1.write('max duration\t' + str(time_list.max()) + '\n')
        f1.write('min duration\t' + str(time_list.min()) + '\n')
        f1.write('mean duration\t' + str(time_list.mean()) + '\n')
        f1.write('variance duration\t' + str(time_list.var()) + '\n')
    plt.show()
