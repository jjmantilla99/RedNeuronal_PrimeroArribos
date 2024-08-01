import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import h5py
#import statistics as stats
import os
import time
#import pandas as pd
#import seaborn as sn
import pickle
from random import randint
from scipy import signal
import pywt
import glob



'''
os.environ['TF_XLA_FLAGS']='--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
'''

def create_model(input_shape, nclases, output_activation, neurons = 680,
                 activation='softsign',dropout_rate=0.1,
                 loss_function = 'mean_squared_error',
                optimizer = 'Adamax', metrics = ['binary_accuracy']):
    """_summary_

    Args:
        input_shape (tuple): Tuple with the input shape of the model
        nclases (int): Number of clases or neurons of the last layer
        neutons (int): NUmber of neuros for multilayer perceptrons leyaer. Defaults to 680
        output_activation : funcion de actibacion de la ultima capa de la red
        activation (str, optional): Activation function of the last layer. Defaults to 'softsign'.
        dropout_rate (float, optional): Droput rate for last layers. Defaults to 0.1.
        loss_function (str, optional): Loss function for the model. Defaults to 'mean_squared_error'.
        optimizer (str, optional): Optimizar for the model. Defaults to 'Adamax'.
        metrics (list, optional): List of metrics for the model. Defaults to ['binary_accuracy'].

    Returns:
        model : Keras model
    """

    input_layer = keras.Input(shape=input_shape)

    neurons1= neurons
    
    initializer='glorot_uniform'#'he_normal'
    
    x = layers.Conv1D(10, 20, strides = 1, padding='same')(input_layer)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling1D(3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dense(3)(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Dense(units = neurons1, kernel_initializer = initializer, activation = activation)(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Dense(units = neurons1, kernel_initializer = initializer, activation = activation)(x)
    classes =layers.Dense(nclases, activation=output_activation)(x)
    
    
    model= keras.Model(input_layer, classes)
    model.summary()
    
    
    model.compile(loss = loss_function,
                  optimizer = optimizer, 
                  metrics = metrics
                  )
    return model

def graficas_metricas(vec_error, error_abs, test_name, dataset, parent_dir, dataset_name):

    n, bins, patches= plt.hist(vec_error,bins=np.histogram_bin_edges(vec_error, bins=26,range=[-1300,1300]), alpha=0.8, rwidth=0.85)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.xlabel('Muestras de Error')
    plt.ylabel('Frecuencia')
    plt.title(test_name)
    plt.ioff()
    plt.savefig(parent_dir +"error_"+dataset+".png")
    plt.close()
    
    n, bins, patches= plt.hist(vec_error,bins=np.histogram_bin_edges(vec_error, bins=80,range=[-400,400]), alpha=0.8, rwidth=0.85)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.xlabel('Muestras de Error - Zoom')
    plt.ylabel('Frecuencia')
    plt.title(test_name)
    plt.ioff()
    plt.savefig(parent_dir +"errorzoom_"+dataset+".png")
    plt.close()  
	
def tranformadas(data, mode = 0):
    """this funtions applies a low pass filter for the signal ang generate num_tf continues
    wavelet transformas

    Args:
        data (numpy array): 3 channel waveform
        mode (int): Mode selector to the type and number of the wavelet transforms, by defaut 0
            Modes: C1 = mean channel 1
                0: [1 channels, 32 tranfors C1] 
                1: [3 channels, 32 tranfors C1, 32 tranfors C2, 32 tranfors C3]
                2: [3 channels, 10 tranfors C1, 10 tranfors C2, 10 tranfors C3]
                3: [3 channels, 50 tranfors C1, 50 tranfors C2, 50 tranfors C3]
                4: [C1, 32 tranfors C1, C2, 32 tranfors C2, C3, 32 tranfors C3]
                5: [C1, 10 tranfors C1, C2, 10 tranfors C2, C3, 10 tranfors C3]
                6: [C1, 50 tranfors C1, C2, 50 tranfors C2, C3, 50 tranfors C3]

    Returns:
        tracesTu: Numpy Array with the channel waveforms and the wavelet transforms.
    """
    
    if mode == 0:
        data = data[:,:,(randint(0,2))]
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        flowpass = signal.filtfilt(b, a, data) #data es la señal a filtrar
    
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data, widths, "morl")
    
        tracesTu=np.zeros((data.shape[0],1251,33))
        tracesTu[:,:,0]=data
        tracesTu[:,:,1]=flowpass
        for w in range(31):
            c=w+2
            tracesTu[:,:,c]=coefs[w]
    if mode == 1:
    
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,32))
        tracesTu_0[:,:,0]=flowpass
        for w in range(31):
            c=w+1
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,1]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,1], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,32))
        tracesTu_1[:,:,0]=flowpass
        for w in range(31):
            c=w+1
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,2]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,2], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,32))
        tracesTu_2[:,:,0]=flowpass
        for w in range(31):
            c=w+1
            tracesTu_2[:,:,c]=coefs[w]
        
        tracesTu = np.concatenate((data,tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
                       
    if mode == 2:
        
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,10))
        tracesTu_0[:,:,0]=flowpass
        for w in range(9):
            c=w+1
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,1]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,1], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,10))
        tracesTu_1[:,:,0]=flowpass
        for w in range(9):
            c=w+1
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,2]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,2], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,10))
        tracesTu_2[:,:,0]=flowpass
        for w in range(9):
            c=w+1
            tracesTu_2[:,:,c]=coefs[w]
        
        tracesTu = np.concatenate((data,tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
        
    if mode == 3:
        
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,50))
        tracesTu_0[:,:,0]=flowpass
        for w in range(49):
            c=w+1
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,1]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,1], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,50))
        tracesTu_1[:,:,0]=flowpass
        for w in range(49):
            c=w+1
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,2]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,2], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,50))
        tracesTu_2[:,:,0]=flowpass
        for w in range(49):
            c=w+1
            tracesTu_2[:,:,c]=coefs[w]
        
        tracesTu = np.concatenate((data,tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
    if mode == 4:

        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,33))
        tracesTu_0[:,:,0]=data[:,:,0]
        tracesTu_0[:,:,1]=flowpass
        for w in range(31):
            c=w+2
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,33))
        tracesTu_1[:,:,0]=data[:,:,0]
        tracesTu_1[:,:,1]=flowpass
        for w in range(31):
            c=w+2
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 32)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,33))
        tracesTu_2[:,:,0]=data[:,:,0]
        tracesTu_2[:,:,1]=flowpass
        for w in range(31):
            c=w+2
            tracesTu_2[:,:,c]=coefs[w]
            
        tracesTu = np.concatenate((tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
        
    if mode == 5:
        
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,11))
        tracesTu_0[:,:,0]=data[:,:,0]
        tracesTu_0[:,:,1]=flowpass
        for w in range(9):
            c=w+2
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,11))
        tracesTu_1[:,:,0]=data[:,:,0]
        tracesTu_1[:,:,1]=flowpass
        for w in range(9):
            c=w+2
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 10)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,11))
        tracesTu_2[:,:,0]=data[:,:,0]
        tracesTu_2[:,:,1]=flowpass
        for w in range(9):
            c=w+2
            tracesTu_2[:,:,c]=coefs[w]
            
        tracesTu = np.concatenate((tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
    if mode == 6: 
        
        b, a = signal.butter(8, 0.3, 'lowpass')   #Configuration filter 8 representa el orden del filtro
        
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_0=np.zeros((data.shape[0],1251,51))
        tracesTu_0[:,:,0]=data[:,:,0]
        tracesTu_0[:,:,1]=flowpass
        for w in range(49):
            c=w+2
            tracesTu_0[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_1=np.zeros((data.shape[0],1251,51))
        tracesTu_1[:,:,0]=data[:,:,0]
        tracesTu_1[:,:,1]=flowpass
        for w in range(49):
            c=w+2
            tracesTu_1[:,:,c]=coefs[w]
            
        flowpass = signal.filtfilt(b, a, data[:,:,0]) #data es la señal a filtrar
        widths = np.arange(1, 50)
        coefs, freqs = pywt.cwt(data[:,:,0], widths, "morl")
        tracesTu_2=np.zeros((data.shape[0],1251,51))
        tracesTu_2[:,:,0]=data[:,:,0]
        tracesTu_2[:,:,1]=flowpass
        for w in range(49):
            c=w+2
            tracesTu_2[:,:,c]=coefs[w]
            
        tracesTu = np.concatenate((tracesTu_0,tracesTu_1,tracesTu_2), axis=2)
        
    return tracesTu
    
def metricas_p(y_pred, targets):
    """
    
    Parameters
    ----------
    y_pred : array
        Array with th predictions returned by the model.predict method.
    targets : array
        Real targets for the data.

    Returns
    -------
    fb_predict1D : TYPE
        DESCRIPTION.
    error_muestra : TYPE
        DESCRIPTION.
    error_muestra_total : TYPE
        DESCRIPTION.
    vec_p : TYPE
        DESCRIPTION.
    vec_p100 : TYPE
        DESCRIPTION.
    tp : TYPE
        DESCRIPTION.
    fn : TYPE
        DESCRIPTION.
    tp100 : TYPE
        DESCRIPTION.
    fn100 : TYPE
        DESCRIPTION.

    """
    fb_predict1D=np.ones((y_pred.shape[0],),dtype=np.float64)
    h0=np.ones((1251,))
    error_muestra = []
    error_muestra_total = []
    vec_p = []
    vec_p100 = []
    fb_h0=np.ones((y_pred.shape[0],1251),dtype=np.float64)
    tp=0
    fn=0
    tp100=0
    fn100=0
    for j in range(y_pred.shape[0]):#RECORRE TODAS LAS TRAZAS
        
        s=(y_pred[j,:])
        for u in range(0, len(s)):
            h0[u]=(s[u]-s[u-1])
            if u==0:
                h0[u]=(s[u]-s[u])
        mxh0=np.max(h0)
        fb_h0[j]=h0
        fb_predict1D[j]=(np.where(h0==mxh0))[0][0]
        error = abs(fb_predict1D[j]-targets[j,0])
        error_total = (fb_predict1D[j]-targets[j,0])
        error_muestra = np.append(error_muestra,error)
        error_muestra_total = np.append(error_muestra_total,error_total)
        if error ==0:
            tp=tp+1
            vec_p = np.append(vec_p, error)
        else:
            fn=fn+1
        if error <= 100:
            tp100=tp100+1
            vec_p100 = np.append(vec_p100, error)
        else:
            fn100=fn100+1
    return fb_predict1D, error_muestra, error_muestra_total, vec_p, vec_p100, tp, fn, tp100, fn100

def metricas_psgc(y_pred, targets):

    fb_predict1D=np.ones((y_pred.shape[0],),dtype=np.float64)
    h0=np.ones((1251,))
    error_muestra = []
    error_muestra_total = []
    vec_p = []
    vec_p100 = []
    fb_h0=np.ones((y_pred.shape[0],1251),dtype=np.float64)
    tp=0
    fn=0
    tp100=0
    fn100=0
    for j in range(y_pred.shape[0]):#RECORRE TODAS LAS TRAZAS
        
        s=(y_pred[j,:])
        for u in range(0, len(s)):
            h0[u]=(s[u]-s[u-1])
            if u==0:
                h0[u]=(s[u]-s[u])
        mxh0=np.max(h0)
        fb_h0[j]=h0
        fb_predict1D[j]=(np.where(h0==mxh0))[0][0]
        error = abs(fb_predict1D[j]-targets[j])
        error_total = (fb_predict1D[j]-targets[j])
        error_muestra = np.append(error_muestra,error)
        error_muestra_total = np.append(error_muestra_total,error_total)
        if error ==0:
            tp=tp+1
            vec_p = np.append(vec_p, error)
        else:
            fn=fn+1
        if error <= 100:
            tp100=tp100+1
            vec_p100 = np.append(vec_p100, error)
        else:
            fn100=fn100+1
    return fb_predict1D, error_muestra, error_muestra_total, vec_p, vec_p100, tp, fn, tp100, fn100

def metricas_n(y_pred, targets):
    
    fp=0
    tn=0
    p_max = []
    for j in range(y_pred.shape[0]):#RECORRE TODAS LAS TRAZAS
        s=(y_pred[j,:])
        ms = max(s)
        p_max = np.append(p_max, ms)
        if ms < 0.6:
            tn=tn+1
        else:
            fp=fp+1
    return p_max, fp, tn

def load_h5(parent_dir,dataset_name):
    """Tload_h5 loads the dataset from a h5 or hdf5 file

    Args:
        parent_dir (string): Path for the parenr directory where is located the dataset
        dataset_name (string): Name of the h5/hdf5 file that has the dataset

    Returns:
        x : Numpy Array with the data
        y : Numpy Array  with the target
        label : Numpy array with the target number of the data
    """
    dataset = h5py.File(parent_dir + dataset_name + '.h5', 'r')
    print('Loading Data')
    x = dataset.get('data')
    x = np.array(x)
    print('Loading Target')
    y = dataset.get('target')
    y = np.array(y)
    print('Loadign Labels')
    label = dataset.get('targets_number')
    label = np.array(label)
    dataset.close()
    x = (x - np.mean(x))/(np.std(x))
    return x, y, label 

def train_model(model, x_train, y_train, x_valid, y_valid, path, train_name, nepochs=60, npatience=3, snum=1, histo_freq=2):   

    callback = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = npatience, verbose=0, mode="auto"),
        tf.keras.callbacks.TensorBoard(log_dir = path + '/logs', 
                                    histogram_freq = histo_freq),
        #tf.keras.callbacks.ModelCheckpoint(filepath = path + '/model_per_epoch/model_epoch_{epoch:02d}.hdf5',
                                        #save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience=10)
        ]
    
    start_time = time.time()
    
    
    history=model.fit(x_train, y_train, validation_data=(x_valid, y_valid) ,epochs=nepochs, callbacks=[callback])
    print("--- %s seconds ---" % (time.time() - start_time))    
  
    print('Creando Graficas Funcions de Perdida y Precision')
    model.save(path +'/model_cnn' + str(snum) + train_name +'.h5')
    np.save(path +'/model_cnn' + str(snum) + train_name +"_history.npy",history.history)
    plt.figure()
    plt.plot(history.history["loss"],label='Pérdida de entrenamiento')
    plt.plot(history.history["val_loss"],label='Pérdida de validación')
    plt.title(train_name + "- Perdidas")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.ioff()
    plt.savefig(path +'/funcionperdida_' +str(snum)+".png",bbox_inches='tight')
    
    plt.figure()
    plt.plot(history.history["binary_accuracy"],label='Accuracy de entrenamiento')
    plt.plot(history.history["val_binary_accuracy"],label='Accuracy de validación')
    plt.title(train_name + '- Accuracy')
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(path +'/acurracy_' + str(snum) + ".png", bbox_inches='tight')
    plt.ioff()
    return model, history, (time.time() - start_time)

def cnn_model(input_shape, nclases, activation='softsign' ):
    '''
    NOT IN USE

    Parameters
    ----------
    input_shape : TYPE
        DESCRIPTION.
    nclases : TYPE
        DESCRIPTION.
    activation : TYPE, optional
        DESCRIPTION. The default is 'softsign'.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    
    input_img = keras.Input(shape=input_shape)
    #nclasses = 1251
    #'sigmoid'
    
    neurons1=680
    
    initializer='glorot_uniform'#'he_normal'
    
    x = layers.Conv1D(10, 20, strides = 1, padding='same')(input_img)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling1D(3, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dense(3)(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    
    x=layers.Flatten()(x)
    x = layers.Dense(units = neurons1, kernel_initializer = initializer, activation = activation)(x)
    x = layers.Dense(units = neurons1, kernel_initializer = initializer, activation = activation)(x)
    classes =layers.Dense(nclases, activation=activation)(x)
    
    
    model= keras.Model(input_img, classes)
    model.summary()
    
    
    model.compile(loss='mean_squared_error',
    #'mean_squared_error',#'mean_absolute_error',#'binary_crossentropy', 
                  optimizer='Adamax', 
                  #metrics=[tf.keras.metrics.Accuracy()]
                  metrics=['binary_accuracy']
                  )
    return model
   
def pred_model(data_parent_dir, dataset_name, model, test_name, predict_dir, snum=1, mode = 0):

    dataset_name = dataset_name

    print('Loading Data for Predictions')
    print('Dataset: ', dataset_name)

    data,_, label = load_h5(data_parent_dir,dataset_name)

    if len(label.shape) == 2:
        label = label[:,0]

    forma = len(data.shape)
    if forma == 4:
        data = data[:,:1251,:,:]
        data = np.reshape(data,(data.shape[0],1251,3))

    data = tranformadas(data, mode = mode) 
    y_pred = model.predict(data)
    
    [vec_pred, error_abs, error_total, errp, errp100, tp, fn, tp100, fn100] = metricas_psgc(y_pred, label)
    metrics = [vec_pred, error_abs, error_total, errp, errp100, tp, fn, tp100, fn100]  
    
    graficas_metricas(vec_error = error_total, 
                      error_abs = error_abs, 
                      test_name = test_name, 
                      dataset = dataset_name, 
                      parent_dir = predict_dir + '/graficas_metricas/', 
                      dataset_name = test_name)
    
    with open(predict_dir +'/data/resultados'+ str(snum) + dataset_name + '.pkl', 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    graph_predictions(10, data, label, y_pred, predict_dir, dataset_name)

    return data, label , y_pred

def pred_model_epoch(predict_dir, data_parent_dir, dataset_name, mode = 0):
    """
    This function load all the model saved durign training, model saved under
    model_per_epoch folder

    Args:
        predict_dir (string): Path where the results of the training are located
        data_parent_dir (string): Path where the datasets are located
        dataset_name (string): Dataset name for the predictions
        mode (int, optional): _description_. Defaults to 0.
    """
    models = glob.glob(predict_dir)

    #epocas = epocas /10
    #epocas = np.array([1,2,3,4,5,6,7,8,9,10]) * epocas
    #parent_dir = 'E:/Proyecto/Database/h5/'
    #dataset_name = '1620_TestNoNoise_STEAD'
    #load data
    data,_, label = load_h5(data_parent_dir,dataset_name)
    if len(label.shape) == 2:
        label = label[:,0]
    forma = len(data.shape)
    if forma == 4:
        data = data[:,:1251,:,:]
        data = np.reshape(data,(data.shape[0],1251,3))
    data = tranformadas(data, mode) 
    index = np.random.randint(low = 0, high = (data.shape[0]-1), size=3)
    for model in models:
        model=keras.models.load_model(model)
        y_pred = model.predict(data)
        path_res = os.path.join(predict_dir, 'model_per_epoch/output_epoch_' + model.split('.')[0])
        try: 
            os.mkdir(path_res) 
        except OSError as error: 
            print(error) 
        graph_pred_static(index, data, label, y_pred, path_res, dataset_name)
    
def graph_predictions(num_graphics, x, label, y_predict, save_path, dataset_name):

    path_graficas = save_path + '/graficas_muestras/'
    
    try:
        print('\n#######')
        os.mkdir(os.path.join(path_graficas, dataset_name)) 
    except OSError as error: 
        print(error) 
    
    index = np.random.randint(low = 0, high = (x.shape[0]-1), size=num_graphics)
    
    for i in index:
    
        y_target = np.zeros(1251)
        y_target[int(label[i]):] = 1
        fig, axs = plt.subplots(5, sharex=True, figsize=(10,4))
        fig.tight_layout()
        axs[0].plot(x[i,:,0])
        axs[0].set_title('Data - Selected Channel' + 'index:' + str(i))
        axs[1].plot(x[i,:,1])
        axs[2].plot(x[i,:,2])
        axs[3].plot(y_target)
        axs[3].set_title('Target')
        axs[4].plot(y_predict[i,:])
        axs[4].set_title('Prediction')
        name = save_path + '/graficas_muestras/' +dataset_name + '/index_' + str(i) + '.png'
        fig.savefig(name)

def graph_pred_static(index, x, label, y_predict, save_path, dataset_name):
    '''
    Parameters
    ----------
    num_graphics: int
        Number of events to graph.
    x : numpy array
        Waveforms data.
    label : numpy array
        Target for the waveforms.
    y_predict : numpy array
        Array with the return from model.predict method.
    save_path : string
        Path to save the figure.

    Returns
    -------
    None.

    '''
    path_graficas = os.path.join(save_path, 'graficas_muestras/')
    
    try: 
        os.mkdir(path_graficas) 
    except OSError as error: 
        print(error) 
        
    path_graficas = os.path.join(save_path, 'graficas_muestras/' + dataset_name)
    
    try: 
        os.mkdir(path_graficas) 
    except OSError as error: 
        print(error) 
    
    for i in index:
    
        y_target = np.zeros(1251)
        y_target[int(label[i]):] = 1
        fig, axs = plt.subplots(5, sharex=True, figsize=(10,4))
        fig.tight_layout()
        axs[0].plot(x[i,:,0])
        axs[0].set_title('Data - Selected Channel' + 'index:' + str(i))
        axs[1].plot(x[i,:,1])
        axs[2].plot(x[i,:,2])
        axs[3].plot(y_target)
        axs[3].set_title('Target')
        axs[4].plot(y_predict[i,:])
        axs[4].set_title('Prediction')
        name = save_path + '/graficas_muestras/' + dataset_name + '/index_' + str(i) + '.png'
        fig.savefig(name)
        plt.show()

def generate_label(data, label, label_shape='gaussian', label_width=30):
    # target = np.zeros(self.Y_shape, dtype=self.dtype)
    target = np.zeros_like(data)

    if label_shape == "gaussian":
        label_window = np.exp(
            -((np.arange(-label_width // 2, label_width // 2 + 1)) ** 2)
            / (2 * (label_width / 5) ** 2)
        )
    elif label_shape == "triangle":
        label_window = 1 - np.abs(
            2 / label_width *
            (np.arange(-label_width // 2, label_width // 2 + 1))
        )
    else:
        print(f"Label shape {label_shape} should be guassian or triangle")
        raise
        
        
    for i in range(data.shape[0]):
        print('###LABEL##')
        print(label[i])
        if int(label_width/2) < label[i] < 1250 - int(label_width/2):
                lim_inf = int(label[i] - label_width/2)
                lim_sup = int(label[i] + label_width/2 + 1)
                print(lim_inf,lim_sup,label[i],lim_sup-lim_inf)
                print('Aqui',time.localtime())
                target[i,lim_inf:lim_sup]=label_window[:int(lim_sup-lim_inf)]
        
        else:
            if lim_sup > 1251:
                lim_sup = 1251
            if lim_inf > 1251:
                lim_inf = 1251
            target[i,lim_inf:lim_sup]=0
            
    return target
