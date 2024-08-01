# **Conjunto de datos provenientes del Catálogo Sísmico Colombiano.**

Este repositorio contiene los archivos necesarios y usados para la creación del conjunto de datos provenientes del Catálogo Sísmico Colombiano y que se encuentran almacenados en el clúster del grupo de investigación en Conectividad y Procesamiento de Señales (CPS) de la Universidad Industrial de Santander.

# Archivo HDF5

Posterior al proceso de descarga, al cruce de información del Servicio Geologico Colombiano, se realizo una verificación de estos datos, que posteriormente fueron organizados en un archivo hdf5 llamado **sgc_dataset.hdf5**, alojado en el clúster del grupo de investigación CPS bajo la ruta /media/hdd/Data/SGC_dataset/sgc_dataset.hdf5, archivo contiene 16203 eventos con información sobre la llegada de ondas P y ondas S, guardados en el grupo llamado **Picados** y 29390 eventos sísmicos sin los cuales se cuenta información sobre la llegada de ondas P y ondas S en el grupo llamado **NoPicados**; todos los eventos sísmicos descargados poseen información sobre tres canales diferentes, los canales grabados están guardados con el siguiente orden HHE, HHN y HHZ. Adicionalmente se poseen 21 meta datos para cada evento sismico tambien almacenados en un archivo Pickle llamado Picks_2018_2022.pkl.

Los eventos para los cuales se tiene información de las Ondas P y Ondas S, fueron etiquetados de la siguiente manera **publicID_station**, correspondiendo a la unión del ID publico generado por el SGC y la estación a la cual pertenece la grabación de las trazas y para el otro grupo, estos etiquetados de la siguiente forma **eventtime_station**, para este segundo caso, eventtime corresponde al metadato de la fecha a la cual sucedió el evento sísmico y la estación que grabo las formas de onda.

## **Tener en cuenta**
-  El archivo **environment.yml** corresponde al environment de anaconda usado para este proceso, para mas información de como crear un environment en anaconda dirigirse a este [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)


## Función **consultaSGC** 

Esta función corresponde al script modificado del profesor Jheyston Serrano, al cual se le removio la interfaz gráfica para poder realizar la ejecución en el cluster del grupo CPS de una forma ás automatica. En la siguiente tabla se listan los parámetros en el orden que fueron declarados y que representan, lo valores por defecto fueron propuestos por el profesor y no fueron modificados y retorna un DataFrame con la consulta realizada.


|     Parámetros    |                         Descripción                        |
|:----------------:|:----------------------------------------------------------:|
|         f        | Path al archivo Excel que contiene la consulta a descargar** |
| TimeRecordBefore |                 Valor por defecto igual a 1                |
|  TimeRecordAfter |                Valor por defecto igual a 10                |
|   MinMagnitude   |  Valor mínimo de la magnitud de su interés, por defecto 1  |
|   MaxMagnitudq   |  Valor máximo de la magnitud de su interés, por defecto 9  |
|     MinDepth     |                Profundidad mínima del evento               |
|     MaxDepth     |                Profundidad maxima del evento               |
|    MinLatitud    |            Latitud mínima de la zona de interés            |
|    MaxLatitud    |            Latitud máxima de la zona de interés            |
|    MinLongitud   |            Longitud mínima de la zona de interés           |
|   MaxLonguitud   |            Longitud máxima de la zona de interés           |

** excel con los eventos que pueden ser descargados desde el [Catálogo Sísmico Colombiano](http://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta_Seiscomp/consultaexperta.php), recordar que a este archivo se le debe borrar el apartado de los parámetros de búsqueda y renombrar las columnas de la siguiente forma:
- Columna A: FECHA  
- Columba B: LATITUD
- Columna C: LONGITUD
- Columna D: PROFUNDIDAD

La siguiente animación aclara lo previamente descrito


![animación](Proceso_descarga_sgc/Imagenes/config_excel.gif )

 

## Función **joint_data**
|  Parametros |                                          Descripción                                          |
|:-----------:|:---------------------------------------------------------------------------------------------:|
| consulta_df |                 DataFrame proveniente de la función consultaSGC                |
|   info_csv  |                        DataFrame con la información proveniente del SGC                       |

Esta función retorno un pandas DataFrame con los eventos y las formas de Onda previamente descargadas por la función **consultaSGC** y agrega con la información faltante por parte del Servicio Geologico.

## Función **copy_publicID**

Con esta función se busca relacionar el canal HHE de un evento sismico, ya que sobre este canal el Servicio Geologico Colombiano no imprime ninguna información.

Recibe como argumento un pandas DataFrame (previamente tratato por la función joint_data), y retorna dos pandas DataFrame:

- El primer pandas DataFrame posee todos los eventos y sus formas de onda con información sobre la llegada de las ondas P y S.

- El segundo pandas DataFrame posee todos los eventos y sus formas de onda que no fueron picados por parte del Servicio Geológico Colombiano, por lo tanto, estos eventos no tiene información sobre la llegada de las ondas P y S.


