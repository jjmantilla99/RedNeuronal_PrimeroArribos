# **Implementación Red CNN para la detección de primero arribos**

En éste directorio se encuentran los archivos python necesarios para realizar los procesos de entrenamiento y predicción sobre la red propuesta para la detección de primeros arribos. Adicionalmente, se encuentra un archivo llamado "env_local.yaml" utilizado en los procesos descritos anteriormente. 


## **Entrenamiento**

Se buscó automatizar el proceso dentrenamiento, para esto a manera de ejemplo basta con ejecutar desde la consola el siguiente comando:

python main.py --data_parent_dir=/media/hdd/Data/Database/ --train_dataset_name=12963_TrainNoNoise_STEAD --valid_dataset_name=1620_ValidNoNoise_STEAD --results_dir=/home/semillero/Juan_Miguel/Fase3/ --wtf_mode=3 --test_name=F3_SGC+STEAD_Dropout30_LRvariable_lw10 --label_width=10

El comando requiere las siguientes especifiaciones:

**--data_parent_dir**: Path donde se encuentran los recortes realizados sobre los datos a usar durante el entrenamiento.

**--train_dataset_name**: Nombre del archivo con los datos destinados al entrenamiento.

**--valid_dataset_name**: Nombre del archivo con los datos destinado para la validación durante el entrenamiento.

**--results_dir**: Path donde se guardaran los resultados de la prueba.

**--wtf_mode**: Define la cantidad de transformadas Wavelet a generar sobre los conjuntos de datos.

**--test_name**: Nombre asignado a la prueba que se ejecuta.

**--label_width**: Define el ancho de un target gaussiano, este parametro es opcional.

**Salida:** Una carpeta con el nombre del la prueba que contiene: el modelo entrenado, las gráficas de pérdida y accuracy en entrenamiento y validación, los logs del entrenamiento y un archivo .npy que contiene información sobre el entrenamiento.


## **Predicción**

En la predicción, se realizó un script para utilizar el modelo entrenado. El archivo model_prediction.py contiene lo mecionado anteriormente. Este se basa en una función que utiliza el modelo cargado con la librería Keras:

    
    Parametros de la función pred_model utilizada.
    ----------
    data_parent_dir : string
        Path donde se encuentran almacenados los archivos H5 que tienen los eventos.
    dataset_name : string
        Nombre del archivo H5 que se va a utilizar.
    model : tensorflow.python.keras.engine.functional.Functional
        Modelo cargado con Keras.
    test_name : string
        Nombre de la prueba.
    snum : int, optional
        Etiqueta para determinar el numero del experimento. Predefinida en 1.
    predict_dir : string, optional
        Directorio donde se desea guardar los resultados. Predefinida en ''.
    mode : int, optional
        Modo de entrada de la red. Predefinida en 0.

    Retorna
    -------
    None.
	
El uso de estas líneas de código genera una carpeta con el nombre de la prueba donde se encuentra: una carpeta data con un archivo .pkl con los resultados, una carpeta con graficas de eventos y una carpeta con diagramas de barras de los resultados obtenidos.






