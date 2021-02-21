import os
from data_loader import loadfile
import tensorflow as tf
# from testing_file import testing
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle
import librosa
from scipy.io import wavfile
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from data_loader import IBM
from validation_denoiseLSTM import Validation
from deepLSTMnetwork import network
max_length =320
sr = 16000

# #Pink Noise Training
data_path = '/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN'
dfc=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/clean-audio')
dfn=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/noisy-audio')
dfc.sort()
dfn.sort()
dfnoise=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/noise')
dfnoise.sort()

global batch_size
nfft = 512
hop_length = 257
batch_size = 64
frame_size = 257
num_hidden = 256
keep_probability = 0.2
num_layers = 1
seq_len = tf.placeholder(tf.int32, None)
keep_pr = tf.placeholder(tf.float32, ())
dim1 = tf.placeholder(tf.int32,None)
q2_x = tf.placeholder(tf.float32, [None, max_length, frame_size, 1])
q2_y = tf.placeholder(tf.float32, [None, max_length, frame_size])
x_abs = tf.placeholder(tf.float32, [None, max_length, frame_size])
s_abs = tf.placeholder(tf.float32, [None, max_length, frame_size])
# q2_x = tf.placeholder(tf.float32, [None, None, frame_size, 1])
# q2_y = tf.placeholder(tf.float32, [None, None, frame_size])
# x_abs = tf.placeholder(tf.float32, [None, None, frame_size])
# s_abs = tf.placeholder(tf.float32, [None, None, frame_size])
with tf.variable_scope("window",reuse=tf.AUTO_REUSE) as scope:

    filters = {
        'wc1': tf.get_variable('W0', shape=(3,3,1,1), initializer=tf.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('W01', shape=(3,3,1,64), initializer=tf.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('W1', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
        'wc4': tf.get_variable('W2', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
        'wc5': tf.get_variable('W3', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer()),
        'wc6': tf.get_variable('W4', shape=(3,3,512,256), initializer=tf.contrib.layers.xavier_initializer()),
        'wc7': tf.get_variable('W5', shape=(3,3,256,128), initializer=tf.contrib.layers.xavier_initializer()),
        'wc8': tf.get_variable('W6', shape=(3,3,128,64), initializer=tf.contrib.layers.xavier_initializer()),
        'wc9': tf.get_variable('W7', shape=(1,1,64,1), initializer=tf.contrib.layers.xavier_initializer()),
    }

#NETWORK

conv1 = tf.nn.conv2d(q2_x,filters['wc1'], strides=(1,2,2,1), dilations = (1,1,1,1),padding="SAME")
#conv2d(tf.pad(conv1,[[0,0],[0,0],[2,2],[0,0]]),
#     filters['wc11'], strides=(1,2,2,1), dilations = (1,1,1,1),padding="VALID")
# pdb.set_trace()
conv1 = tf.keras.layers.BatchNormalization()(conv1)
conv1 = tf.nn.leaky_relu(conv1)
conv2 = tf.nn.conv2d(conv1,filters['wc2'], strides=(1,2,2,1), dilations = (1,1,1,1),padding="SAME")
conv2 = tf.keras.layers.BatchNormalization()(conv2)
conv2 = tf.nn.leaky_relu(conv2)
conv3 = tf.nn.conv2d(conv2,filters['wc3'], strides=(1,2,2,1), dilations = (1,1,1,1),padding="SAME")
conv3 = tf.keras.layers.BatchNormalization()(conv3)
conv3 = tf.nn.leaky_relu(conv3)
conv4 = tf.nn.conv2d(conv3,filters['wc4'], strides=(1,2,2,1), dilations = (1,1,1,1),padding="SAME")
conv4 = tf.keras.layers.BatchNormalization()(conv4)
conv4 = tf.nn.leaky_relu(conv4)
# pdb.set_trace()
# conv4 = tf.reshape(conv4,[batch_size,9,10*256])

# def get_a_cell(lstm_size, keep_prob):
#     lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
#     drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_probability)
#     return drop


# with tf.name_scope('lstm'):
#  cell = tf.nn.rnn_cell.MultiRNNCell(
#  [get_a_cell(10*256, keep_probability) for _ in range(num_layers)]
#  )

# lstm4,_=tf.nn.dynamic_rnn(cell,conv4,sequence_length=seq_len,dtype=tf.float32)

# lstm4 = tf.reshape(lstm4,[batch_size,10,9,256])
conv4 = tf.keras.layers.BatchNormalization()(conv4)
conv5 = tf.keras.layers.Conv2DTranspose(filters = 128 , kernel_size = (3,3), strides = (2,2),activation = 'selu',padding="SAME")(conv4)
# pdb.set_trace()
conv5 = tf.slice(conv5, [0,0,0,0],[-1,-1,33,128])
# print(conv3.shape,conv5.shape)
cat5 = tf.concat([conv3,conv5],axis=3)
conv5 = tf.keras.layers.BatchNormalization()(conv5)
conv6 = tf.keras.layers.Conv2DTranspose(filters = 64 , kernel_size = (3,3), strides = (2,2),activation = 'selu',padding="SAME")(cat5)
conv6 = tf.slice(conv6, [0,0,0,0],[-1,-1,65,64])
cat6 = tf.concat([conv2,conv6],axis=3)
cat6 = tf.keras.layers.BatchNormalization()(cat6)
conv7 = tf.keras.layers.Conv2DTranspose(filters = 1 , kernel_size = (3,3), strides =(2,2),activation = 'selu',padding="SAME")(cat6)
conv7 = tf.slice(conv7, [0,0,0,0],[-1,-1,129,1])
cat7 = tf.concat([conv1,conv7],axis=3)
cat7 = tf.keras.layers.BatchNormalization()(cat7)
conv8 = tf.keras.layers.Conv2DTranspose(filters = 1 , kernel_size = (3,3), strides = (2,2),padding="SAME",activation = 'selu')(cat7) # strides [2,2] initially
conv8 = tf.slice(conv8, [0,0,0,0],[-1,-1,257,1])
pdb.set_trace()
# pdb.set_trace()
# print(conv8.shape)
conv8 = tf.reshape(conv8,[-1,dim1,257])
out = tf.math.sigmoid(conv8)
# print(conv8.shape)
# conv8 = tf.nn.conv2d(cat7,filters['wc8'],strides=(1,10
# pdb.set_trace()
# def get_a_cell(lstm_size, keep_prob):
#     lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
#     drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_probability)
#     return drop


# with tf.name_scope('lstm'):
#  cell = tf.nn.rnn_cell.MultiRNNCell(
#  [get_a_cell(num_hidden, keep_probability) for _ in range(num_layers)]
#  )

# output,_=tf.nn.dynamic_rnn(cell,conv8,sequence_length=seq_len,dtype=tf.float32)
# rnn_out = tf.layers.dense(output, 257, kernel_initializer=
#     tf.contrib.layers.xavier_initializer(),activation = None)
# pdb.set_trace()
# fin_out = tf.sigmoid(conv8)

dim = seq_len[0]
lr = 0.0001
# cost = tf.reduce_mean(tf.losses.mean_squared_error(conv8[:, :max_length,:257],
#     s_abs)) + tf.reduce_mean(tf.losses.mean_squared_error(out[:, :max_length,:257],
#     q2_y[:, :max_length, :257]))
cost=tf.reduce_mean(tf.losses.mean_squared_error(out[:, :max_length,:257],
    q2_y[:, :max_length, :257]))
# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate= lr).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate = lr, 
    # momentum = 0.95, use_nesterov = True).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()


# #TRAINING LOOP        
init=tf.global_variables_initializer()
sess.run(init)
epochs = 400
error = np.zeros(epochs)
for epoch in range(epochs):
    random = np.arange(0, 10000-batch_size-1,batch_size)
    np.random.shuffle(random)
    for i in range(len(random)):
        #print(i,len(random))
        start = int(random[i])
        end = int(start + batch_size)
        dfcc=dfc[start:end]
        dfnn=dfn[start:end]
        dfnoisee=dfnoise[start:end]
        dfcc.sort()
        dfnn.sort()
        dfnoisee.sort()
        trx, X, X_abs, X_len = loadfile(data_path,dfnn,'noisy-audio',nfft,hop_length,flag = 0) 
        trs, S, S_abs, S_len = loadfile(data_path,dfcc,'clean-audio',nfft,hop_length,flag = 0)
        trn, N, N_abs, N_len = loadfile(data_path,dfnoisee,'noise',nfft,hop_length,flag = 0)
        M = IBM(S_abs, N_abs)
        # print(M)
        epoch_y = np.array(M).swapaxes(1,2)
        epoch_x = np.array(X_abs).swapaxes(1,2)
        # epoch_x = np.reshape(epoch_x,(batch_size,300,257))
        epoch_x = np.reshape(epoch_x,(batch_size,max_length,257,1))
        seqlen = np.array(X_len)
        # print(seqlen,q2_y.shape)
        # print(epoch_x.shape)
        l, _,C8= sess.run([cost, optimizer,conv8], feed_dict = {q2_x: epoch_x, q2_y: epoch_y,
        x_abs : np.array(X_abs).swapaxes(1,2),
        s_abs: np.array(S_abs).swapaxes(1,2), seq_len: max_length, keep_pr: 1, dim1:max_length})
        # print(C8.shape,epoch_y.shape)
        error[epoch] += l
        # saver.save(sess,'q2model/new_data_apple/CRNN_stft')
    saver.save(sess,'q2model/new_data_apple/after_epoch/CLSTMNN_wind_IITGN')
                # saver.save(sess,'model_iter',global_step=i)
    #print(SNR)
    print('Epoch', epoch+1, 'completed out of ', epochs,'; loss: ', error[epoch])

saver.save(sess,'q2model/new_data_apple/combined_loss/CLSTMNN_wind_IITGN')


#TESTING
# with tf.Session() as sess:
#     # new_saver = tf.train.import_meta_graph("/home/perrryosa/Speech-Denoising-With-RNN/q2model/my_rnn_model.meta")
#     new_saver=tf.train.Saver(max_to_keep=0)
#     # new_saver.restore(sess, tf.train.latest_checkpoint('./'))

#     # new_saver.restore(sess,tf.train.latest_checkpoint('/home/perrryosa/Speech-Denoising-With-RNN/result multilayer/q2model/'))

#     new_saver.restore(sess,
#         "/home/perrryosa/Speech-Denoising-With-RNN/q2model/new_data_apple/after_epoch/CLSTMNN_wind_IITGN")
#     print_tensors_in_checkpoint_file(
#         "/home/perrryosa/Speech-Denoising-With-RNN/q2model/new_data_apple/after_epoch/CLSTMNN_wind_IITGN",
#          all_tensors=False, tensor_name='')


#     # pdb.set_trace()
#     data_path = '/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/test_wind'
#     #dftest=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/new_data_apple/validation')
#     dftest=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/test_wind/noisy')
#     dftest.sort()
#     tex, TEX, TEX_abs, TEX_len = loadfile(data_path,dftest, 'noisy', flag = 1)

#     def test_SNR(M_pred, X, i):
#         M_pred = 1 * (M_pred > 0.5)
#         M_pred = M_pred.T
#         print(M_pred)
#         S_pred = M_pred * X
#         print(S_pred[100,100])
#         # print(S_pred.shape)
#         s_pred = librosa.istft(S_pred, win_length = 256, hop_length = 129)
#         # s_pred = tf.signal.inverse_stft(tf.transpose(S_pred),frame_length = 512, frame_step = 257, fft_length = 512)
#         print('writing audios...')
#         # s_pred = s_pred.eval(session=tf.compat.v1.Session())
#         # print(s_pred.shape)

#         # librosa.output.write_wav('/home/perrryosa/Speech-Denoising-With-RNN/QANTAS-1-Channel/recovered/recovered'
#         #  + str(i) + '.wav', s_pred, 16000)
#         # print(s_pred.shape,type(s_pred))
#         wavfile.write('/home/perrryosa/Speech-Denoising-With-RNN/test_result/denoised'
#          + str(i) + '.wav', 16000, s_pred)
#         # librosa.output.write_wav('/home/perrryosa/Speech-Denoising-With-RNN/test_result/denoised'+str(i)+'.wav', s_pred, 16000)


#     #Getting predictions for all test sets
#     for i in range(len(TEX_abs)):
#         print(i)
#         epoch_x = np.zeros((1, TEX_abs[i].shape[1], TEX_abs[i].shape[0]))
#         epoch_y = np.zeros((1, TEX_abs[i].shape[1], TEX_abs[i].shape[0]))
#         epoch_x[0,:,:] = TEX_abs[i].T
#         # epoch_x[0,:,:] = TEX_abs[i].eval(session=tf.compat.v1.Session()).T
#         # print(tf.transpose(TEX_abs[i]).shape,type(tf.transpose(TEX_abs[i])))
#         # epoch_x[0,:,:] = tf.transpose(TEX_abs[i])
#         # print(epoch_x)
#         # print(epoch_x.shape,"as")
#         epoch_x = np.reshape(epoch_x,(1,TEX_len[i],129,1))
#         # print(TEX_abs[i].shape[0])
#         # print(epoch_x.shape,"ass")
#         TEM_pred= sess.run(conv8, feed_dict = {q2_x:epoch_x, seq_len : TEX_len[i] ,keep_pr: 1, dim1:TEX_abs[i].shape[1] })
#         print(TEM_pred,TEM_pred[0][100,100])
#         test_SNR(TEM_pred[0,:,:], TEX[i], i)


sess.close()