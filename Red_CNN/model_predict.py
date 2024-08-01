# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:29:17 2022

@author: super
"""
from tensorflow import keras
from fbsismology import pred_model

    


data_parent_dir = 'F:/Proyecto/Database/h5/'


test_name = 'STEAD_Dropout30_LRvariable_wtfmode1'
snum=1
mode = 1
parent_dir = 'F:/Proyect_tests/'


model=keras.models.load_model("C:/UIS/Proyecto/ResultadosFases/Fase1_new/model_cnn1"+test_name+".h5")        
                                 
predict_dir = parent_dir + '/' + test_name


dataset_name = '1620_TestNoNoise_STEAD'
data, label, y_pred = pred_model(data_parent_dir = data_parent_dir,
           dataset_name = dataset_name,
           model = model,
           test_name = test_name,
           snum = snum,
           predict_dir = predict_dir,
           mode = mode)

dataset_name = 'TestSGC'
data, label, y_pred = pred_model(data_parent_dir = data_parent_dir,
           dataset_name = dataset_name,
           model = model,
           test_name = test_name,
           snum = snum,
           predict_dir = predict_dir,
           mode = mode)

