import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.sparse import csr_matrix, hstack
import scipy
import random
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import keras
from keras.layers import Input, Dense, Lambda, Reshape, Permute, Dropout, CuDNNGRU, Embedding
from keras.models import Model, load_model
from keras.models import Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras.losses import categorical_crossentropy
import gc

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

batch_size = 100
original_dim = 784
latent_dim = 2 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 100#int(xy_dims/2)
epochs = 10
num_classes = 10

y_train = pd.read_csv('rsc15_4_train_tr.txt',sep='\t')
y_test = pd.read_csv('rsc15_4_test.txt',sep='\t')
y_val = pd.read_csv('rsc15_4_train_valid.txt',sep='\t')
x_train = pd.read_csv('rsc15_1_train_tr.txt',sep='\t')
x_test = pd.read_csv('rsc15_1_test.txt',sep='\t')

# Data dimantion
ytrain_dims = len(y_train['ItemId'].unique()) 
ytest_dims = len(y_test['ItemId'].unique())
xtrain_dims = len(x_train['ItemId'].unique()) # 拿全部的item_id來做mapper 
xtest_dims = len(x_test['ItemId'].unique()) 

# 把 y_train x_train 併在一起做

xy_train = np.concatenate((x_train['ItemId'].unique(),y_train['ItemId'].unique()))
xy_train = np.unique(xy_train)
xy_dims = len(xy_train)

# mapping ,然後 to_categorical\

item_ids = xy_train
item2idx = pd.Series(data=np.arange(xy_dims), index=item_ids)
itemmap = pd.DataFrame({'ItemId':item_ids,'item_idx':item2idx[item_ids].values})

# 處裡 y_train中的缺失值
y_trainnna =  item2idx[y_train['ItemId']].dropna()[0:400000]
# 處裡 x_train中的缺失值
x_trainnna =  item2idx[x_train].dropna()[0:400000]
# 處裡 x_test中的缺失值
x_testnna =  item2idx[x_test['ItemId']].dropna()[0:2799]
# 處裡 y_test中的缺失值
y_testnna =  item2idx[y_test['ItemId']].dropna()[0:2799]

# y_train = to_categorical(y_trainnna, num_classes=xy_dims)
y_test = to_categorical(y_testnna, num_classes=xy_dims)
# x_train = to_categorical(x_trainnna, num_classes=xy_dims)
x_test = to_categorical(x_testnna, num_classes=xy_dims)
x_trainnna = to_categorical(x_trainnna, num_classes=xy_dims)
y_trainnna = to_categorical(y_trainnna, num_classes=xy_dims)

# Build CVAE model

x = Input(shape=(xy_dims,)) # original_dim shape=(xy_dims,)
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

y = Input(shape=(xy_dims,),name='y') # 输入类别 shape=(xy_dims,)
h2 = Dense(intermediate_dim, activation='relu',name='h2')(y)
y_mean = Dense(latent_dim,name='y_mean')(h2) # 这里就是直接构建每个类别的均值
y_var = Dense(latent_dim,name='y_var')(h2)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(xy_dims, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model([x, y], [x_decoded_mean, y_mean, y_var])

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_log_var - y_var - K.square(z_mean - y_mean) 
                        - K.square(K.sqrt(K.exp(z_log_var)) 
                                   - K.sqrt(K.exp(y_var))), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# generator if need
def generate_data_from_pd(x,batch_size):
    xlen = len(x)
    i = 0
    for u in range(0,12000):
        i = random.randint(0,1200000)
        start = i * batch_size
        end = (i+1) * batch_size
        if end > xlen:
            end = xlen
            start = xlen - 100
        x_batch = x[start:end]
#         y_batch = y[start:end]
        xin = to_categorical(x_batch, num_classes=xy_dims)
#         yin = to_categorical(y_batch, num_classes=xy_dims)
        yield xin
        
def generate_data(x,y,batch_size):
    xlen = len(x)
    for j in range(10):
        a = generate_data_from_pd(x,batch_size)
        b = generate_data_from_pd(y,batch_size)
        for i in range(12000): #xlen/batch_size+1
            input1 = a.__next__()
            input2 = b.__next__()
#             input1 = input1.reshape(-1,9029)
#             input2 = input2.reshape(-1,9029)
            yield [{'input_1':input1,'input_2':input2},None]
    #         c = np.array([input1,input2])
    #         yield c.reshape(9029,2)
    #     return a,b
    #     return [a,b]

# fit_generator(memory problem)
# generator_X = generate_data_from_pd(list(x_trainnna),1)
# generator_Y = generate_data_from_pd(list(y_trainnna),1)
# b = generate_data(list(y_trainnna))
c = generate_data(list(x_trainnna),list(y_trainnna),100)
# d = generate_data(list(y_testnna))
vae.fit_generator(generator=c,
                 steps_per_epoch=12000,#len(y_trainnna)/batch_size+1
                 epochs=10,
                 verbose=1,
                 validation_data=([x_test, y_test], None),
                 validation_steps=len(y_test))

# fit(main method)
vae.fit([x_trainnna, y_trainnna], 
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None))


# save original model
vae.save('./vae_12240.h5')

# remove y_input
vae_model = vae
vae_model.summary()

last_layer = vae_model.layers[-3]
lambda_layer = vae_model.layers[-5]

vae_model.layers.pop()
vae_model.layers.pop()
vae_model.layers.pop()
vae_model.layers.pop()
vae_model.layers.pop()
vae_model.layers.pop()

vae_model.summary()

model_to_train = Model(inputs=vae_model.input, outputs=vae_model.get_output_at(0)[0])

model_to_train.summary()

model_to_train.save('./new_vae_12240.h5')

# remove x_input

new_vae = model_to_train

x0 = new_vae.layers[0]
x1 = new_vae.layers[1]
x2 = new_vae.layers[2]
x3 = new_vae.layers[3]
x4 = new_vae.layers[4]
x5 = new_vae.layers[5]
x6 = new_vae.layers[6]

# 把 new_vae 的 input 抽掉
y = x1
y1 = x2(y.get_output_at(0))
y2 = x3(y.get_output_at(0))
y = x4([y1, y2])
y = x5(y)
y = x6(y)

noInput_vae = Model(input=x1.input, output=[y])

# save the final embedding model
noInput_vae.save('./noInput_vae_12240.h5')