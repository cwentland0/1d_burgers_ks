# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 00:48:52 2019

@author: ashis
"""


from keras.models import load_model
import keras
import math
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
from keras.layers import Conv1D,MaxPooling1D,Flatten,Reshape,LeakyReLU,UpSampling1D,BatchNormalization
from keras.activations import relu,hard_sigmoid,elu,softplus,softsign,exponential,selu,tanh,linear
import numpy as np
import scipy.io as sio
import numpy as np
import sklearn.metrics
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import scipy

def decoder_jacobian(fom_vector,code_current):
    
    
    sess = tf.InteractiveSession()
    sess.run()
    
    #fom_vector = scaling(fom_vector,mean,minimum,maximum)
    #code_current = encoder.predict(fom_vector,batch_size=1)
    inputTensor = autoencoder.layers[-1].get_input_at(0)
    outputTensor = autoencoder.layers[-1].get_output_at(0)
    jac = np.zeros((fom_vector.shape[1],code_current.shape[1]))
    
    
    for i in range(fom_vector.shape[1]):
    
        print('spatial dimension:',i)
        outputTensor = autoencoder.layers[-1].get_output_at(0)[:,i]
        grad_func = tf.gradients(outputTensor,inputTensor)[0]
        gradients = sess.run(grad_func, feed_dict={inputTensor: code_current})
        jac[i,:] = gradients
    
    sess.close()
    
    return jac

def scaling_snapshots(training):
    
    nt = training.shape[0]
    nx = training.shape[1]
    mean = np.mean(training,axis=0)
    mean_assembled = np.repeat(mean.reshape((nx,1)),nt,axis=1)
    affine = training-mean_assembled.T
    minimum = np.amin(affine)
    maximum = np.amax(affine)
    affine = (affine-minimum)/(maximum-minimum)
    
    return affine,mean.reshape((nx,1)),minimum,maximum


def scaling(x,mean,minimum,maximum):
    
    nt = x.shape[0] 
    nx = x.shape[1]
    mean_assembled = np.repeat(mean.reshape((nx,1)),nt,axis=1)
    affine = x-mean_assembled.T
    affine = (affine-minimum)/(maximum-minimum)
    
    return affine

def prolongate(x,mean,minimum,maximum):
    
    nt = x.shape[0] 
    nx = x.shape[1]
    mean_assembled = np.repeat(mean.reshape((nx,1)),nt,axis=1)
     
    x = x*(maximum-minimum)
    x = x+minimum+mean_assembled.T 
    
    return x
    
    
mat_contents = sio.loadmat('solf1.mat')
xFOM = mat_contents['solf1']            #full order solution , should be of form smaples x lat.dims

#mat_contents = sio.loadmat('solf2.mat')
#xFOM = mat_contents['solnFOM']            #full order solution , should be of form smaples x lat.dims

mat_contents = sio.loadmat('u0.mat')
u0 = mat_contents['u0']  #initial condition

mat_contents = sio.loadmat('A.mat')
A = mat_contents['A']    #flux calculating matrix

#mat_contents = sio.loadmat('x.mat')
#xx = mat_contents['x'] #spatial vector

mat_contents = sio.loadmat('res.mat')
res_FOM = mat_contents['res']

mat_contents = sio.loadmat('res_wt.mat')
res_FOM_wt = mat_contents['res_wt']

mat_contents = sio.loadmat('dt.mat')
dt = mat_contents['dt']

#xFOM_scaled,mean_FOM,minimum_FOM,maximum_FOM = scaling_snapshots(xFOM)
#setting up training matrix
u0_assembled = np.repeat(u0.reshape((1,xFOM.shape[1])),xFOM.shape[0],axis=0)
training = xFOM-u0_assembled
#training = xFOM
#performing scaling 
training,mean,minimum,maximum = scaling_snapshots(training)
res_FOM_scaled,mean_res,minimum_res,maximum_res = scaling_snapshots(res_FOM)
training1 = training[250:500,:]
training = training[0:250,:]
#training = res_FOM_scaled
#training = prolongate(training,mean,minimum,maximum)
input_shape = training.shape
nt = input_shape[0]
nx = input_shape[1]
#callbacks
cs = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=300, verbose=1, mode='min', baseline=None, restore_best_weights=True)
#encoder
input = Input(shape=(input_shape[1],))

x = Reshape(target_shape=(input_shape[1],1))(input)

#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv1D(8, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = MaxPooling1D(pool_size=2, strides=1, padding='same', data_format='channels_last')(x)


#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv1D(16, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = MaxPooling1D(pool_size=2, strides=1, padding='same', data_format='channels_last')(x)

#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv1D(32, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = MaxPooling1D(pool_size=2, strides=1, padding='same', data_format='channels_last')(x)

#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv1D(64, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = MaxPooling1D(pool_size=2, strides=1, padding='same', data_format='channels_last')(x)

#x = Flatten()(x)

#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
#x = Dense(256, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = Dense(20, activation=elu, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
encoder = models.Model(input,x)
#encoder.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
encoder.compile(optimizer=optimizers.adam(lr=1e-3), loss='mean_squared_error')
output_shape = encoder.layers[-1].output_shape    
shape_before_flatten = encoder.layers[-2].input_shape
shape_after_flatten = encoder.layers[-2].output_shape
#decoder
input = Input(batch_shape=output_shape)
#x = Dense(256, activation=elu, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
#x = Dense(shape_after_flatten[1],activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
#x = Reshape(target_shape=(shape_before_flatten[1],shape_before_flatten[2]))(x)

#x = UpSampling1D(size=1)(x)
x = Conv1D(32, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(input)
#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

#x = UpSampling1D(size=1)(x)
x = Conv1D(16, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

#x = UpSampling1D(size=1)(x)
x = Conv1D(8, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
#x = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

#x = UpSampling1D(size=1)(x)
x = Conv1D(1, 2, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=linear, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)

x = Reshape(target_shape=(input_shape[1],))(x)
decoder = models.Model(input,x)
#decoder.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
decoder.compile(optimizer=optimizers.adam(lr=1e-3), loss='mean_squared_error')
    
input = Input(batch_shape=input_shape)
autoencoder = models.Model(input,decoder(encoder(input)))
autoencoder.compile(optimizer=optimizers.adam(lr=1e-3), loss='mean_squared_error')
#autoencoder.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# learning schedule callback

# Fit the model
#history=autoencoder.fit(training, training, epochs=2000, batch_size=input_shape[0], callbacks=callbacks_list, verbose=1)
history = autoencoder.fit(x=training, y=training1,  epochs=1, batch_size=input_shape[0], callbacks=[cs],shuffle=False)
current_loss = history.history['loss']
plt.figure()
plt.plot(current_loss, color='blue')
plt.yscale('log')
plt.show()
#saving models
#encoder.save('encoder.h5')
#autoencoder.save('autoencoder.h5')
#decoder.save('decoder.h5')

weights_list = autoencoder.layers[2].get_weights()

nx = 256

n_dense = 1
n_conv = 4

#W_dense = weights_list[0].T

#for i in (1,n_dense-1):
    
#    W_dense = weights_list[i].T@W_dense
    
input_filters = [64,32,16,8]    
output_filters = [32,16,8,1]

#W_conv = W_dense

for i in range(1):
    
    W=weights_list[i]
    
    
    for k in range(output_filters[i]):
        kernel = W[:,0,k]
        w_h_stacked = scipy.sparse.diags([kernel[0]*np.ones(nx),kernel[1]*np.ones(nx-1)],[0,1])
        for l in range(1,input_filters[i]):
            
            kernel = W[:,l,k]
            temp = scipy.sparse.diags([kernel[0]*np.ones(nx),kernel[1]*np.ones(nx-1)],[0,1])
            w_h_stacked = scipy.sparse.hstack([w_h_stacked,temp])
            
        if k==0:
            w_v_stacked = w_h_stacked
        else:
            w_v_stacked = scipy.sparse.vstack([w_v_stacked,w_h_stacked])
                
    W_conv = w_v_stacked

for i in range(1,4):
    
    W=weights_list[i]
    
    
    for k in range(output_filters[i]):
        kernel = W[:,0,k]
        w_h_stacked = scipy.sparse.diags([kernel[0]*np.ones(nx),kernel[1]*np.ones(nx-1)],[0,1])
        for l in range(1,input_filters[i]):
            
            kernel = W[:,l,k]
            temp = scipy.sparse.diags([kernel[0]*np.ones(nx),kernel[1]*np.ones(nx-1)],[0,1])
            w_h_stacked = scipy.sparse.hstack([w_h_stacked,temp])
            
        if k==0:
            w_v_stacked = w_h_stacked
        else:
            w_v_stacked = scipy.sparse.vstack([w_v_stacked,w_h_stacked])
                
    W_conv = w_v_stacked@W_conv
                
W_final = W_conv

code = encoder.predict(training,batch_size=256)
#recon_w = decoder.predict(code,batch_size=250)
recon = decoder.predict(code,batch_size=250)
code1 = np.zeros((250,16384))
for i in range(64):
    code1[:,i*256:(i+1)*256]=code[:,:,i]
    

recon_w = W_final@code1.T
recon_w = recon_w.T


