import os
import numpy as np
import librosa
from scipy.io import wavfile
import tensorflow as tf
global max_length
max_length = 300
def loadfile(data_path,df,str_dir,n,h,batch_size,flag=0):
    list_tr = []
    list_stft = []
    list_stft_abs = []

    list_length = []
    
    for i in range(batch_size):
        sr,s=wavfile.read(data_path+'/'+str_dir+'/'+df[i])        
        if (flag == 1):
            list_tr.append(s)
        #Calculating STFT
        s=s.astype(float)
        stft = librosa.stft(s, n_fft= n, hop_length= h, center = False)
        # stft = tf.transpose(tf.signal.stft(s,frame_length = 512, frame_step =257, fft_length = 512 , pad_end = True))
        # print(stft.shape)
        #Appending STFT to list
        # list_stft.append(stft)
        stft_len = stft.shape[1]    
        if flag == 0:            
            if stft_len<max_length:
                stft = np.pad(stft, ((0,0),(0, max_length-stft_len)), 'constant')
            else:
                stft=stft[:,:max_length]
            list_stft.append(stft)
        # _ , phase = librosa.magphase(stft)

        #Calculating Absolute of STFT
        # print(stft.shape)
        if (flag == 1): 
            # stft = np.pad(stft, ((0,0),(0, 16-stft.shape[1]%16)), 'constant')
            print(stft.shape, 'fuck-intended')
            list_stft.append(stft)
        # list_stft.append(stft)
        stft_len = stft.shape[1]        
        stft_abs = np.abs(stft)
        list_stft_abs.append(stft_abs)
        
        #Appending time-length of STFT to list
        list_length.append(stft_len)
    return list_tr, list_stft, list_stft_abs, list_length


# def IBM(S, N):
#     M = []

#     for i in range(len(S)):
#         m_ibm = 1 * (S[i] > N[i])
#         M.append(m_ibm)
#     return M

def IBM(S,N):# Ideal Ratio mask
    M=[]

    for i in range(len(S)):
        m_irm = (S[i]**2 / (S[i]**2 + N[i]**2 +1e-10))
        M.append(m_irm)

    return M
