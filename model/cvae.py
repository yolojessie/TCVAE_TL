import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.sparse import csr_matrix, hstack
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda, Reshape, Permute, Dropout, CuDNNGRU, Embedding
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras.losses import categorical_crossentropy
import gc


batch_size = 100
original_dim = 784
latent_dim = 2 # 隐变量取2维只是为了方便后面画图
intermediate_dim = original_dim/2
epochs = 10
num_classes = 10


y_train = pd.read_csv('rsc15_3_f60p_train_tr.txt',sep='\t')
y_test = pd.read_csv('rsc15_3_f60p_test.txt',sep='\t')
x_train = pd.read_csv('rsc15_1_train_tr.txt',sep='\t')
x_test = pd.read_csv('rsc15_1_test.txt',sep='\t')


# Data dimantion
ytrain_dims = len(y_train['ItemId'].unique()) 
ytest_dims = len(y_test['ItemId'].unique())
xtrain_dims = len(x_train['ItemId'].unique())
xtest_dims = len(x_test['ItemId'].unique()) 

# 把 y_train x_train 併在一起做

xy_train = np.concatenate((x_train['ItemId'].unique(),y_train['ItemId'].unique()))
xy_train = np.unique(xy_train)
xy_dims = len(xy_train)

# todo
# mapping ,然後 to_categorical\

item_ids = xy_train
item2idx = pd.Series(data=np.arange(xy_dims), index=item_ids)
itemmap = pd.DataFrame({'ItemId':item_ids,'item_idx':item2idx[item_ids].values})

del xy_train
gc.collect()

# 處裡 x_train中的缺失值
x_trainnna =  item2idx[x_train['ItemId']].dropna()[0:398482]
# 處裡 x_test中的缺失值
x_testnna =  item2idx[x_test['ItemId']].dropna()# 處裡 y_test中的缺失值
y_testnna =  item2idx[y_test['ItemId']].dropna()[0:4589]

y_train = to_categorical(item2idx[y_train['ItemId']], num_classes=xy_dims)
y_test = to_categorical(y_testnna, num_classes=xy_dims)
x_train = to_categorical(x_trainnna, num_classes=xy_dims)
x_test = to_categorical(x_testnna, num_classes=xy_dims)

x = Input(shape=(xy_dims,)) # original_dim shape=(xy_dims,)
h = Dense(latent_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

y = Input(shape=(xy_dims,)) # 输入类别 shape=(xy_dims,)
y_mean = Dense(latent_dim)(y) # 这里就是直接构建每个类别的均值

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(latent_dim, activation='relu')
decoder_mean = Dense(xy_dims, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model([x, y], [x_decoded_mean, y_mean])

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean - y_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit([x_train, y_train], 
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None))

vae.save('./vae_12360_II.h5')


