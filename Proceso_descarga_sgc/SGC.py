from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS
from obspy.core import UTCDateTime, trace
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, figure, subplot, plot
from mpl_toolkits import mplot3d
from math import log
from shapely.geometry import Point
import geopandas
from obspy import read
from tqdm import tqdm
import smtplib
from email.message import EmailMessage


def consultaSGC(f, TimeRecordBefore, TimeRecordAfter, MinMagnitud, MaxMagnitud, MinDepth, MaxDepth, MinLatitud,
                MaxLatitud, MinLongitud, MaxLongitud):
    """
    Esta funcion realiza la consulta al Catalogo Sísmico Colombiano y realiza la descarga de las trazas sismicas que se
    encuentran en la ruta especificada por f

    Args:
        f: Ruta del archivo Excel que se quiere realizar una consulta
        TimeRecordBefore: Asignar 1 como defecto
        TimeRecordAfter: Asignar 10 como valor por defecto
        MinMagnitud: Valor de la magnitud minima que se quiere realizar en la consulta
        MaxMagnitud: Valor de la magnitud maxima que se quiere realizar en la consulta
        MinDepth: Valor minimo de la profundidad  minima para realizar la consulta de las trazas
        MaxDepth: Valor maximo de la profundidad  minima para realizar la consulta de las traza
        MinLatitud: Valor de la latitud minima de la zona de interes de la consulta.
        MaxLatitud: Valor de la latitud maxima de la zona de interes de la consulta.
        MinLongitud: Valor de la longitud minima de la zona de interes de la consulta.
        MaxLongitud: alor de la longitud maxima de la zona de interes de la consulta.

    Returns:
        points_df: Pandas Dataframe con la información y trazas sismiscas descargadas desde el Catalogo
        Sísmico Colombiano

    """

    global listaStrems, data

    timeRecordBefore = 60 * int(TimeRecordBefore)  # en segundos
    timeRecordAfter = 60 * int(TimeRecordAfter)  # en segundos
    minMagnitud = float(MinMagnitud)  # 5.0
    maxMagnitud = float(MaxMagnitud)  # 5.8
    canalBusqueda = 'HH?'
    minDepthBusqueda = float(MinDepth)
    maxDepthBusqueda = float(MaxDepth)

    minLatitudBusqueda = float(MinLatitud)
    maxLatitudBusqueda = float(MaxLatitud)
    minLongitudBusqueda = float(MinLongitud)
    maxLongitudBusqueda = float(MaxLongitud)

    #print("\nInformacion ingresada:")
    #print("minMagnitud:{}".format(minMagnitud))
    #print("maxMagnitud:{}".format(maxMagnitud))
    #print("timeRecordBefore:{}".format(timeRecordBefore))
    #print("timeRecordAfter:{}".format(timeRecordAfter))
    #print("canalBusqueda:{}".format(canalBusqueda))
    #print("minDepthBusqueda:{}".format(minDepthBusqueda))
    #print("maxDepthBusqueda:{}".format(maxDepthBusqueda))
    #print("minLatitudBusqueda:{}".format(minLatitudBusqueda))
    #print("maxLatitudBusqueda:{}".format(maxLatitudBusqueda))
    #print("minLongitudBusqueda:{}".format(minLongitudBusqueda))
    #print("maxLongitudBusqueda:{}\n".format(maxLongitudBusqueda))

    try:
        # Lectura archivo excel
        data = pd.read_excel(f)
        data2 = data.copy()
        #print(data.head())
        #print('aqui')

        if 'HORA_UTC' in data:
            data2["FECHA"] = data["FECHA"] + " " + data["HORA_UTC"]
        else:
            data2["FECHA"] = data["FECHA"]

        data2 = data[
            (data['MAGNITUD'] >= minMagnitud) &
            (data['MAGNITUD'] <= maxMagnitud) &
            (data['PROFUNDIDAD'] >= minDepthBusqueda) &
            (data['PROFUNDIDAD'] <= maxDepthBusqueda) &
            (data['LATITUD'] >= minLatitudBusqueda) &
            (data['LATITUD'] <= maxLatitudBusqueda) &
            (data['LONGITUD'] >= minLongitudBusqueda) &
            (data['LONGITUD'] <= maxLongitudBusqueda)]

        #print(data)
        #print(data2)

        zdata = data2['PROFUNDIDAD'].to_numpy()
        xdata = data2['LONGITUD'].to_numpy()
        magdata = data2['MAGNITUD'].to_numpy()
        ydata = data2['LATITUD'].to_numpy()

        #print(data2['LATITUD'].describe())
        #print(data2['LONGITUD'].describe())
        #print(data2['PROFUNDIDAD'].describe())
        #print(data2['MAGNITUD'].describe())

    except:
        # messagebox.showerror(message="No se encontraron eventos", title="Error")
        print('ERROR - No se encontraron eventos')
        return None

    listNetwork = []
    listStation = []
    listLocation = []
    listStartTime = []
    listEndTime = []
    listChannel = []
    listSampling_rate = []
    listDelta = []
    listNpts = []
    listEvenTime = []
    listXEventdata = []
    listZEventdata = []
    listYEventdata = []
    listMagEventdata = []
    listLatitudStation = []
    listLongitudStation = []
    listData = []
    listaStrems = []

    # Filtrar estaciones -- path Excel estaciones.
    # dataEstaciones = pd.read_excel('/media/hdd/Data/SGC/Estaciones/estaciones.xlsx')
    dataEstaciones = pd.read_excel('/media/hdd/Data/sgc_v2/Estaciones/estaciones.xlsx')
    if (canalBusqueda == "BH?" or canalBusqueda == "HH?"):
        df_mask = dataEstaciones[
            (dataEstaciones[
                 'Localizador'] == 0) &  # El localizador esta definido en 0 pero podria ser 10 para un acelerometro
            (dataEstaciones['Latitud'] >= minLatitudBusqueda) &
            (dataEstaciones['Latitud'] <= maxLatitudBusqueda) &
            (dataEstaciones['Longitud'] >= minLongitudBusqueda) &
            (dataEstaciones['Longitud'] <= maxLongitudBusqueda) &
            ((dataEstaciones['Canal'] == canalBusqueda[:-1] + "E") |
             (dataEstaciones['Canal'] == canalBusqueda[:-1] + "N") |
             (dataEstaciones['Canal'] == canalBusqueda[:-1] + "Z"))]

    else:
        df_mask = dataEstaciones[
            (dataEstaciones[
                 'Localizador'] == 0) &  # El localizador esta definido en 0 pero podria ser 10 para un acelerometro
            (dataEstaciones['Latitud'] >= minLatitudBusqueda) &
            (dataEstaciones['Latitud'] <= maxLatitudBusqueda) &
            (dataEstaciones['Longitud'] >= minLongitudBusqueda) &
            (dataEstaciones['Longitud'] <= maxLongitudBusqueda) &
            (dataEstaciones['Canal'] == canalBusqueda)]

    # print(df_mask['Estacion'])
    strEstaciones = ""
    setEstaciones = set()

    for strDfEsta in df_mask['Estacion']:
        strEstaciones += strDfEsta + ","
        setEstaciones.add(strDfEsta)

    strEstaciones = strEstaciones[:-1]
    str_val = ','.join(list(map(str, setEstaciones)))
    print("Estaciones encontradas en la zona seleccionada:")
    print(str_val)
    print("found %s event(s):" % len(data2))
    # print(data['FECHA'])
    #print(data2.columns.values.tolist())

    for i in tqdm(data2.index):
        # for i in tqdm(range(2)):

        # print(data2['FECHA'][i])
        strTime = data2['FECHA'][i].replace(" ", "T")
        eventTime = UTCDateTime(strTime)
        startTime = eventTime - timeRecordBefore
        endTime = eventTime + timeRecordAfter

        zdata = data2['PROFUNDIDAD'][i]
        xdata = data2['LONGITUD'][i]
        magdata = data2['MAGNITUD'][i]
        ydata = data2['LATITUD'][i]

        # print (strTime)
        # print (str(startTime)[:-1])
        # print (str(endTime)[:-1])

        # LLamado al URL de la base de datos del SGC
        URL = "http://sismo.sgc.gov.co:8080/fdsnws/dataselect/1/query?starttime=" + str(startTime)[
                                                                                    :-1] + "&endtime=" + str(endTime)[
                                                                                                         :-1] + "&network=CM&sta=" + str_val + "&cha=" + canalBusqueda + "&loc=00&format=miniseed&nodata=404"
        # URL = "http://sismo.sgc.gov.co:8080/fdsnws/dataselect/1/query?starttime="+str(startTime)[:-1]+"&endtime="+str(endTime)[:-1]+"&network=CM&sta=*&cha=HHZ&loc=*&format=miniseed&nodata=404"

        try:
            st = read(URL)
            # print(str_val)
            # print(st.__str__(extended=True))

            for stream in st:
                # print(stream.stats.station)
                listNetwork.append(stream.stats.network)
                listStation.append(stream.stats.station)

                df2 = df_mask[df_mask['Estacion'] == str(stream.stats.station)]

                for lat in df2['Latitud']:
                    latitudStation = lat

                for lon in df2['Longitud']:
                    longitudStation = lon

                listLatitudStation.append(latitudStation)
                listLongitudStation.append(longitudStation)

                listLocation.append(stream.stats.location)

                listChannel.append(stream.stats.channel)
                listStartTime.append(stream.stats.starttime)
                listEndTime.append(stream.stats.endtime)
                listSampling_rate.append(stream.stats.sampling_rate)
                listDelta.append(stream.stats.delta)
                listNpts.append(stream.stats.npts)
                listEvenTime.append(eventTime)
                listZEventdata.append(zdata)
                listXEventdata.append(xdata)
                listYEventdata.append(ydata)
                listMagEventdata.append(magdata)
                listData.append(stream.data)
                listaStrems.append(stream)
                # print(listaStrems)
                # print(listData)
                # print(stream.data)





        except:
            # print(st)
            print("\nRegistro no encontrado\n")

    ########################################

    points_df = pd.DataFrame(list(zip(listNetwork, listStation, listLatitudStation, listLongitudStation,
                                      listLocation, listChannel,
                                      listStartTime, listEndTime,
                                      listSampling_rate, listDelta, listNpts,
                                      listEvenTime, listZEventdata, listXEventdata, listYEventdata, listMagEventdata,
                                      listData)),
                             columns=['Network', 'Station', 'LatStation', 'LongStation', 'Location', 'Channel',
                                      'Starttime', 'Endtime', 'Samplingrate', 'Delta', 'Npts', 'Eventime', 'DepthEvent',
                                      'LongEvent', 'LatEvent', 'MagEvent', 'Data'])

    #print(points_df['LatStation'].describe())
    #print(points_df.head())
    #print("Se finalizo la consulta")

    return points_df

def joint_data(consulta_df, info_csv):
    """
    Realiza la union de la información proveniente del SGC, su principal funcion es unir los datos
    y guardarlos en un archivo pickle, pero retorna es el numero de filas a la cuales se les añadio
    la información.

    Args:
        consulta_df: dataframe proveniente de consultasgc
        info_csv: dataframe cargado partiendo del csv por parte del SGC

    Returns:
        new_data: pandas DataFrame con la información faltante por parte del Servicio Geologico
    """

    consulta_df['Eventime'] = consulta_df['Eventime'].apply(lambda x: x.isoformat())

    imp_columns = ['event_time_value', 'pick_time_value', 'waveformID_channelCode', 'waveformID_stationCode',
                   'publicID', 'phase_code', 'pick_time_value', 'time_value_ms']

    new_data = consulta_df.join(info_csv[imp_columns].set_index(['event_time_value', 'waveformID_stationCode', 'waveformID_channelCode']),
                on=['Eventime', 'Station', 'Channel'], lsuffix='sgc', rsuffix='csv')

    # picados = new_data[(new_data.phase_code == 'P') | (new_data.phase_code == 'S')].shape[0]
    # new_data.to_pickle(path)

    return new_data

def make_trace(data):
    """
    Crea un objeto trace del modulo Obspy a partir de un fila (pandas series) de un dataframe con toda la información
    previamente añadida

    Parameters
    ----------
    data :  Una fila de un dataframe con toda la información añadida - Un Pandas Series

    Returns
    -------
    trace : Objeto trace de Obspy

    """
    tr = trace.Trace(data.Data)
    # Add the correct stats for the trace
    tr.stats.network = data.Network
    tr.stats.location = data.Location
    tr.stats.channel = data.Channel
    tr.stats.station = data.Station
    tr.stats.starttime = UTCDateTime(data.Starttime)
    tr.stats.sampling_rate = data.Samplingrate
    tr.stats.npts = data.Npts
    tr.stats.phase_code = data.phase_code
    if data.pick_time_value != 'No_Value':
        try:
            tr.stats.phase_time = UTCDateTime(data.pick_time_value)
        except:
            pass
    return tr

def graph_trace(trace):
    """
    Esta funcion se encarga de graficar la traza y sobreponer una linea vertical sobre el picado que se encuentre en la
    propiedad trace.phase_time
    Parameters
    ----------
    trace : Objeto trace de Obspy

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(trace.times("matplotlib"), trace.data, "b-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.axvline(trace.stats.phase_time, color='r', linestyle='--', lw=2,
               label='Onda {}'.format(trace.stats.phase_code))
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.show()

def send_mail(subject, info):
    """
    Envia un email
    Args:
        subject: Subject of the mail
        info: The information from the mail

    Returns:
        None - Just send an email
    """
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = 'mvarbla9825@gmail.com'
    msg['To'] = 'miguel2171521@correo.uis.edu.co,miguelvarbla@gmail.com'

    msg.set_content(info)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('mvarbla9825@gmail.com', 'ttziodfxclltwnls')
        smtp.send_message(msg)

    print('Correo enviado')

def group_by_station(dict_keys):
    """This function groups the data of an event

    Args:
        dict_keys (list): List with the key from the tuple created from an event

    Returns:
        grouped: List with the tuples keys grouped
    """
    stations = list(set([x for _, x, _ in dict_keys]))


    stationsDict = {x: [] for x in stations}

    for x in dict_keys:
        stationsDict[x[1]].append(x)

    grouped = list(stationsDict.values())
    return grouped

def copy_publicID(df):
    """ This funtion copy th publicID attribute to the HHE channel in the dataframe

    Args:
        df (pandas.DataFrame): DataFrame with all the information.

    Returns:
        with_id: pandas.DataFrame that have only events with publicID and picked information
        without_id: pandas.DataFrame with event that doesn't have informatio about the first peacking
    """

    df['publicID'] = df['publicID'].fillna('NoID')
    df['pick_time_value'] = df['pick_time_value'].fillna('No_Value')
    df['phase_code'] = df['phase_code'].fillna('No_phase')
    df['time_value_ms'] = df['time_value_ms'].fillna('No_Value')
    df['Starttime']= df['Starttime'].apply(lambda x: x.isoformat())
    df['Endtime']= df['Endtime'].apply(lambda x: x.isoformat())

    without_id = df[(df.MagEvent >= 2) & (df.phase_code != 'No_phase')]
    eventtime_list = np.unique(without_id.Eventime.to_list())

    without_id = df[df['Eventime'].isin(eventtime_list)]
    without_id = without_id.set_index(['Eventime', 'Station'])

    withid = df[(df.MagEvent >=2) & (df.phase_code != 'No_phase')]
    withid = withid.set_index(['Eventime','Station'])

    eventtimes = withid.index.get_level_values(0).to_list()
    stations = withid.index.get_level_values(1).to_list()

    for event, station in zip(eventtimes, stations):

        public_id = withid.loc[event, station].publicID[0]
        row = without_id.loc[event, station]
        row.publicID = public_id
        without_id.loc[event,station] = row

    with_id = without_id[without_id.publicID != 'NoID']
    without_id = without_id[without_id.publicID == 'NoID']

    return with_id, without_id

def save_with_id(df,standard,no_standard,path):
    """This funciton save the data with NO information about the firts peacking in the hdf5 file

    Args:
        df (pandas.DataFrame): pandas.DataFrame whith No information about the first peacking
        standard (hdf5 group): This needs to be a hdf5 group with this route /no_picked/standard
        no_standard (hdf5 group): This needs to be a hdf5 group with this route /no_picked/no_standard
        path (string): Path to save the csv file with the metadata of the events.

    Returns:
        none
    """

    dates = np.unique(df.index.get_level_values(0))
    no_guardados_no_standard = []
    no_guardados_standard = []

    for date in dates:

        event = df.loc[date, :].groupby(['publicID', 'Station', 'Channel']).Data
        event_dict = dict(tuple(event))
        dict_keys = list(event_dict.keys())
        groups = group_by_station(dict_keys)

        for group in groups:

                event_data = []
                id = group[0][0]
                sta = group[0][1]
                name = '_'.join([id, sta])

                
                for i in range(len(group)):

                    data = event_dict[group[i]].to_numpy()[0]
                    event_data.append(data)
                
                max_points = max(map(len, event_data))

                for i in range(len(event_data)):

                    if event_data[i].size != max_points:

                        num_of_points = (max_points - event_data[i].size) * -1
                        event_data[i] = np.append(event_data[i], event_data[i][num_of_points:])

                if len(event_data) != 3:
                    try:
                        no_standard.create_dataset(name, data=event_data)
                    except:
                        no_guardados_no_standard.append(name)
                else:
                    try:
                        standard.create_dataset(name, data=event_data)
                    except:
                        no_guardados_standard.append(name)

    path = path + 'id.csv'
    df.drop(['Data'], axis=1).reset_index().to_csv(path, index=False)

    return no_guardados_standard, no_guardados_no_standard

def save_without_id(df,standard,no_standard,path):
    """This funciton save the data with information about the firts peacking in the hdf5 file

    Args:
        df (pandas.DataFrame): pandas.DataFrame whit the first peaking information.
        standard (hdf5 group): This needs to be a hdf5 group with this route /picked/standard
        hdf_grupo2 (hdf5 group): This needs to be a hdf5 group with this route /picked/no_standard
        path (string): Path to save the csv file with the metadata of the events.

    Returns:
        none
    """

    df = df.reset_index()

    dates = np.unique(df.Eventime.to_list())

    no_save_standard = []
    no_save_no_standard = []

    
    for date in dates:
        
        event = df[df.Eventime == date].groupby(['Eventime','Station','Channel']).Data
        event = dict(tuple(event))
        groups = group_by_station(event.keys())
        
        for group in groups:
            event_data = []
            event_date = group[0][0]
            event_station = group[0][1]
            name = '_'.join([event_date, event_station])
            
            for i in range(len(group)):
                data = event[group[i]].to_numpy()[0]
                event_data.append(data)

            max_points = max(map(len, event_data))

            for i in range(len(event_data)):

                if event_data[i].size != max_points:

                    num_of_points = (max_points - event_data[i].size) * -1
                    event_data[i] = np.append(event_data[i], event_data[i][num_of_points:])
            
            event_data = np.array(event_data)
            
            if len(event_data) > 3:
                try:
                    no_standard.create_dataset(name, data= event_data)
                except:
                    no_save_no_standard.append(name)
            else:
                try:
                    standard.create_dataset(name, data= event_data)
                except:
                    no_save_standard.append(name)

    path = path + 'no_id.csv'
    df.drop(['Data'], axis=1).reset_index().to_csv(path, index=False)

    return no_save_standard, no_save_no_standard
    
def cut_window(waveform,p_sample,s_sample, mode):
    
    """
    Cut window of 6000 or 1251 samples from the waveform(np.array)
    Args:
        waveform (numpy.array shape=(3,)): waveform numpy.array shape=(3,)
        p_sample (numpy.int64): P sample index
        s_sample (numpy.int64): S sample index
        mode (0 or 1): mode 0 for PhaseNet, mode 1 for 1251 samples

    Returns:
        waverform : Window from waveform with 6000 samples numpy.array shape=(3,6000)
        p_target : New p target/sample numpy.int64
        s_target : New s target/sample numpy.int64
    """

    if mode == 0:
        diff = 6000 - (s_sample - p_sample)
        if diff > 0:
            star_window = p_sample - round(diff/2)
            end_window = s_sample + round(diff/2)
            p_target = p_sample - star_window
            s_target = s_sample - star_window
            t0 = waveform[0][star_window : end_window]
            t1 = waveform[1][star_window : end_window]
            t2 = waveform[2][star_window : end_window]
            new_waveform = np.array([t0,t1,t2])
            #plt.plot(new_waveform.T)
            #plt.axvline(p_target,color='g')
            #plt.axvline(s_target,color='r')
            #print('Normal')

        else:
            # diff es menor que 0, quiere decir que P y S estan muy separados
            # En este caso se corta la ventana a partir de P
            star_window = p_sample - np.random.randint(0,1700)
            p_target = p_sample - star_window
            t0 = waveform[0][star_window : star_window+300]
            t1 = waveform[1][star_window : star_window+300]
            t2 = waveform[2][star_window : star_window+300]
            new_waveform = np.array([t0,t1,t2])
            #plt.plot(new_waveform.T)
            #plt.axvline(p_target)
            #print('Sin S')

        return new_waveform, p_target, s_target
    
    if mode == 1:
        #For Silvia's Network

        shift = np.random.randint(200,1100)
        star_window = p_sample - shift
        end_window = star_window + 1251
        #print(end_window-star_window)
        
        fixed_sample = p_sample - star_window

        t0 = waveform[0][star_window : end_window]
        t1 = waveform[1][star_window : end_window]
        t2 = waveform[2][star_window : end_window]
        new_waveform = np.array([t0,t1,t2])
        #plt.plot(new_waveform.T)
        #plt.axvline(fixed_sample)

        return new_waveform, fixed_sample 
