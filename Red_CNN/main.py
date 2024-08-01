import argparse
from fbsismology import load_h5, tranformadas, create_model, train_model, pred_model
import numpy as np
import os, time

os.environ['TF_XLA_FLAGS']='--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_parent_dir', help='Path de los datasets')
    parser.add_argument('--train_dataset_name', help='Dataset Name for training')
    parser.add_argument('--valid_dataset_name', help='Dataset Name for Validation during training')
    parser.add_argument('--results_dir', help='Path para almacenar los resultados')
    parser.add_argument('--wtf_mode', default = 2,
                         type=int, help='Modo para la función transformadas')
    parser.add_argument('--test_name', help='Nombre Asignado a la prueba')
    parser.add_argument('--pred1_dataset_name',help='Dataset 1 para realizar predicciones')
    parser.add_argument('--pred2_dataset_name',help='Dataset 2 para realizar predicciones')
    args = parser.parse_args()
    return args

def load_data(data_parent_dir, train_dataset_name, valid_dataset_name, mode = 0):

    x_train_st, y_train_st, _ = load_h5(data_parent_dir, train_dataset_name)
    x_valid_st, y_valid_st, _ = load_h5(data_parent_dir, valid_dataset_name)
    
    x_train_st = x_train_st[:2436,:6000,:,:]
    y_train_st = y_train_st[:2436,:6000,:]
    x_valid_st = x_valid_st[:812,:6000,:,:]
    y_valid_st = y_valid_st[:812,:6000,:]
    
    x_train_st = np.reshape(x_train_st,(x_train_st.shape[0],6000,3))
    y_train_st = np.reshape(y_train_st,(y_train_st.shape[0],6000))
    x_valid_st = np.reshape(x_valid_st,(x_valid_st.shape[0],6000,3))
    y_valid_st = np.reshape(y_valid_st,(y_valid_st.shape[0],6000))
    
    assert x_train_st.shape == (12963, 6000, 3) 
    assert y_train_st.shape == (12963, 6000) 
    assert x_valid_st.shape == (1620, 6000, 3) 
    assert y_valid_st.shape == (1620, 6000) 
    
    print('\nGenerando transformadas')
    x_train_st = tranformadas(x_train_st, mode = mode) 
    x_valid_st = tranformadas(x_valid_st, mode = mode) 

    print('\nCargando SGC')

    x_train, y_train, _ = load_h5(data_parent_dir,'TrainSGC')
    x_valid, y_valid, _ = load_h5(data_parent_dir,'ValidSGC')
    
    
    assert x_train.shape == (2436, 6000, 3) 
    assert y_train.shape == (2436, 6000) 
    assert x_valid.shape == (812, 6000, 3) 
    assert y_valid.shape == (812, 6000) 
    
    x_train = tranformadas(x_train, mode = mode) 
    x_valid = tranformadas(x_valid, mode = mode) 
    
    
    x_train = np.concatenate((x_train,x_train_st))
    del x_train_st
    y_train = np.concatenate((y_train,y_train_st))
    del y_train_st
    x_valid = np.concatenate((x_valid,x_valid_st))
    del x_valid_st
    y_valid = np.concatenate((y_valid,y_valid_st))
    
    
    return x_train, y_train, x_valid, y_valid

def main(args):

    # from args
    data_parent_dir = args.data_parent_dir
    train_dataset_name = args.train_dataset_name
    valid_dataset_name = args.valid_dataset_name

    print('#############')
    print('CARGANDO DATOS')

    [x_train, y_train, x_valid, y_valid] = load_data(data_parent_dir,
                                        train_dataset_name=train_dataset_name,
                                        valid_dataset_name=valid_dataset_name,
                                        mode = args.wtf_mode)

    print(x_train.shape,x_valid.shape)
    #assert x_train.shape == (25926, 1251, 99) 
    #assert x_valid.shape == (1620*2, 1251, 99)

    print('DATOS CARGADOS')

    input_shape = (6000, x_train.shape[2])
    model = create_model(input_shape= input_shape, nclases=6000, dropout_rate=0.3)
    print('Se creo el modelo')

    # from args
    test_name = args.test_name
    snum=1
    results_dir = args.results_dir

    # Create Folders
    t = time.strftime("_%H_%M_%S", time.localtime())    
    directory = 'train_' + test_name + t
    os_path = os.path.join(results_dir, directory)
    path = results_dir + directory

    try: 
        os.mkdir(os_path)
        os.mkdir(os.path.join(path,'graficas_metricas'))
        os.mkdir(os.path.join(path,'data'))
        os.mkdir(os.path.join(path, 'graficas_muestras'))
    except OSError as error: 
        print(error)


    model, history, _ = train_model(model = model,
                                            x_train = x_train,
                                            y_train = y_train,
                                            x_valid = x_valid,
                                            y_valid = y_valid,
                                            path = path,
                                            train_name = test_name,
                                            nepochs = 100,
                                            npatience = 50,
                                            snum = snum,
                                            histo_freq=2)
                                                                         

    del x_train, y_train, x_valid, y_valid


    # from args
    #pred_dataset_name = '1620_TestNoNoise_STEAD'

    
if __name__ == '__main__':
    args = read_args()
    main(args)
