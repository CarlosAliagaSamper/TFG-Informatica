""" Modulo auxiliar de datos.

Aqui se implementa toda la funcionalidad relacionada con datos, tanto de demanda energetica
como de temperatura. Tambien se implementa la funcionalidad de normalizacion.

Para su correcto funcionamiento es necesario rellenar la variable token (linea 22)
con su token de e-sios.
"""

# Imports necesarios

import requests
import datetime
from meteostat import Point, Hourly
import pandas as pd
import json


# Variables de configuracion

# Token para la realizacion de peticiones a la API de esios
token = 'INTRODUZCA AQUI SU TOKEN'

# URL para la peticion de indicadores
url_indicadores = 'https://api.esios.ree.es/indicators/'
# Cabeceras necesarias para la peticion de indicadores
headers_indicadores = {
    'Accept': 'application/json; application/vnd.esios-api-v1+json',
    'Content-Type': 'application/json',
    'x-api-key': token,
}

# Dias de la semana
semana = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
# Dia de la semana que fue el 1 de enero de 2024
d_1_1_2024 = 0
# Dia de la semana del primer dia de las peticiones
k = (d_1_1_2024 - 1095) % 7
if k < 0:
    k = k + 7
primer_dia = k

# Horas
horas = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00',
         '06:00:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00',
         '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00',
         '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']

# Dias en los que se cambio la hora de verano a invierno
cambios_hora = ['2023-10-29', '2022-10-30', '2021-10-31']
# Variable de control de los cambios de hora
flag_cambios_hora = False

# Festivos
festivos_nacionales = ['2021-01-01', '2021-01-06', '2021-04-02', '2021-05-01', '2021-10-12',
                       '2021-11-01', '2021-12-06', '2021-12-08', '2021-12-25', '2022-01-01',
                       '2022-01-06', '2022-04-15', '2022-08-15', '2022-10-12', '2022-11-01',
                       '2022-12-06', '2022-12-08', '2023-01-06', '2023-04-07', '2023-05-01',
                       '2023-08-15', '2023-10-12', '2023-11-01', '2023-12-06', '2023-12-08',
                       '2023-12-25']

# Cantidad de datos a usar en los estudios
n_est = 24 * 365 * 2

# Dias de cada mes
# No se tiene en cuenta anyos bisiestos ya que en la franja 2021-2023
# no hay ninguno, pero para la extension de este trabajo habria que tenerlos
# en cuenta
meses = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# Funciones

def peticion_indicadores():
    """Realiza una peticion a la api de esios especifica sobre los indicadores.
    Posteriormente los guarda en el fichero indicadores.txt.
    """
    resp = requests.get(url_indicadores, headers=headers_indicadores)
    f = open("indicadores.txt", "w")
    f.write(resp.text)
    f.close()


def realiza_peticion_esios(indicador, fecha_inicio, fecha_fin):
    """Realiza una peticion a la api de esios sobre un indicador.

    Argumentos:
    indicador -- indicador del tipo de peticion
    fecha_inicio -- primera fecha de la peticion
    fecha_fin -- ultima fecha de la peticion
    """
    url = 'https://api.esios.ree.es/indicators/%s?start_date=%sT00:00:00Z&end_date=%sT23:00:00Z' % (indicador, fecha_inicio, fecha_fin)
    headers = {
        'Accept': 'application/json; application/vnd.esios-api-v1+json',
        'Content-Type': 'application/json',
        'x-api-key': token,
    }
    resp = requests.get(url, headers=headers)
    return resp.json()


def realiza_peticion_temperatura(init_total, end_total):
    """Realiza una peticion a la API de meteostat sobre temperatura en Madrid.

    Argumentos:
    init_total -- fecha de inicio de la peticion
    end_total -- fecha de fin de la peticion
    """
    start_temp = datetime.datetime(init_total.year, init_total.month, init_total.day, 0)
    end_temp = datetime.datetime(end_total.year, end_total.month, end_total.day, 23)
    madrid = Point(40.4165, -3.70256)
    temp = Hourly(madrid, start_temp, end_temp, timezone='UTC')
    temp = temp.fetch()
    return temp['temp']


def arregla_cambio_hora(datetime_utc):
    """Transforma un string con la informacion sobre datetime_utc en caso necesario por cambio de hora

    Argumentos:
    datetime_utc -- datetime_utc devuelto por esios
    """
    anyo = int(datetime_utc[:4])
    dia = int(datetime_utc[8:10])
    mes = int(datetime_utc[5:7])
    hora = datetime_utc[11:19]

    if (dia in cambios_hora) and (hora == '00:00:00'):
        if flag_cambios_hora:
            hora = '01:00:00'
            flag_cambios_hora = False
        else:
            flag_cambios_hora = True
    else:
        flag_cambios_hora = False

    if mes < 10:
        mes_str = "0{}".format(mes)
    else:
        mes_str = "{}".format(mes)

    if dia < 10:
        dia_str = "0{}".format(dia)
    else:
        dia_str = "{}".format(dia)

    bueno = "{}-{}-{}T{}".format(anyo, mes_str, dia_str, hora)

    return bueno


def string_to_datetime(dia_str, hora_str):
    """Transforma una string en datetime

    Argumentos:
    dia_str -- dia en formato string
    hora_str -- hora en formato string
    """
    date_format = '%Y-%m-%d %H:%M:%S'
    diaaux = dia_str + " " + hora_str
    return datetime.datetime.strptime(diaaux, date_format)


def une_datos(demanda_real, demanda_prevision, temp):
    """Une los datos de demanda energetica y temperatura por fecha y hora.

    Argumentos:
    demanda_real -- datos de la demanda real de energia electrica
    demanda_prevision -- datos de la prevision de demanda de energia
    temp -- datos sobre la temperatura
    """
    anterior = demanda_real[0]['indicator']['values'][0]['datetime_utc'][:10]
    actual = ''

    l = 0
    k = primer_dia

    valores_real = []
    valores_prevision = []
    dias = []
    dias_semana = []
    horas2 = []
    temperaturas = []
    fechas = []
    indices = []

    for j in range(0, len(demanda_real)):
        for i in range(0, len(demanda_real[j]['indicator']['values'])):
            fila_real = demanda_real[j]['indicator']['values'][i]
            if fila_real['datetime_utc'][14:16] == '00':
                fila_prevision = demanda_prevision[j]['indicator']['values'][i]

                valores_real.append(fila_real['value']/1000.0)
                valores_prevision.append(fila_prevision['value']/1000.0)

                datetime_utc = arregla_cambio_hora(fila_real['datetime_utc'])
                actual = datetime_utc[:10]

                dias.append(actual)

                if actual != anterior:
                    k = (k + 1) % 7
                    anterior = actual
                
                dias_semana.append(semana[k])
                horas2.append(datetime_utc[11:19])
                temperaturas.append(temp[l])
                fechas.append(string_to_datetime(datetime_utc[:10], datetime_utc[11:19]))
                indices.append(l)

                l = l + 1
    
    tabla = pd.DataFrame({'Indice': indices, 
                          'Valor': valores_real, 
                          'Prevision': valores_prevision,
                          'Dia': dias, 
                          'Dia_Semana': dias_semana, 
                          'Hora': horas2, 
                          'Temperatura': temperaturas, 
                          'Fecha': fechas})
    
    return tabla


def get_rho(dia, dia_semana):
    """Consigue la fraccion de poblacion trabajando un dia.

    Argumentos:
    dia -- dia sobre el que se quiere la fraccion
    dia_semana -- dia de la semana del dia
    """
    f = open("normalizacion/festivos.json")
    festivos = json.load(f)
    f.close()
    rhoW = 1.0
    if dia_semana == 'D':
        return 0.0
    for key in festivos.keys():
        if dia in festivos[key][1]:
            rhoW = rhoW - festivos[key][0]
    return rhoW


def total_laborable(dia, dia_semana):
    """Consigue si un dia es completamente laborable.

    Argumentos:
    dia -- dia sobre el que se quiere la fraccion
    dia_semana -- dia de la semana del dia
    """
    return get_rho(dia, dia_semana) == 1.0


def total_festivo(dia, dia_semana):
    """Consigue si un dia es completamente festivo

    Argumentos:
    dia -- dia sobre el que se quiere la fraccion
    dia_semana -- dia de la semana del dia
    """
    if dia_semana == 'D':
        return True
    return dia in festivos_nacionales


def guarda_datos_normalizacion(data):
    """Guarda los datos necesarios para todos los procesos de normalizacion
    y desnormalizacion.

    Argumentos:
    data -- dataframe con todos los datos de las peticiones
    """
    laborables_aux = dict()
    festivos_aux = dict()
    sabados_aux = dict()
    for diccionario in [laborables_aux, festivos_aux, sabados_aux]:
        diccionario['Valor'] = []
        diccionario['Hora'] = []
    
    for i in range(n_est):
        if total_laborable(data['Dia'][i], data['Dia_Semana'][i]) and data['Dia_Semana'][i] != 'S':
            laborables_aux['Valor'].append(data['Valor'][i])
            laborables_aux['Hora'].append(data['Hora'][i])
        elif total_festivo(data['Dia'][i], data['Dia_Semana'][i]):
            festivos_aux['Valor'].append(data['Valor'][i])
            festivos_aux['Hora'].append(data['Hora'][i])
        elif total_laborable(data['Dia'][i], data['Dia_Semana'][i]) and data['Dia_Semana'][i] == 'S':
            sabados_aux['Valor'].append(data['Valor'][i])
            sabados_aux['Hora'].append(data['Hora'][i])
    
    tabla_laborable = pd.DataFrame({'Valor': laborables_aux['Valor'], 'Hora': laborables_aux['Hora']})
    tabla_festivo = pd.DataFrame({'Valor': festivos_aux['Valor'], 'Hora': festivos_aux['Hora']})
    tabla_sabado = pd.DataFrame({'Valor': sabados_aux['Valor'], 'Hora': sabados_aux['Hora']})

    datos_laborables = dict()
    datos_festivos = dict()
    datos_sabados = dict()

    for hora in horas:
        datos_laborables[hora] = tabla_laborable.loc[tabla_laborable['Hora'] == hora]
        datos_festivos[hora] = tabla_festivo.loc[tabla_festivo['Hora'] == hora]
        datos_sabados[hora] = tabla_sabado.loc[tabla_sabado['Hora'] == hora]
    
    medias_laborables = dict()
    medias_festivos = dict()
    medias_sabados = dict()

    for hora in horas:
        medias_laborables[hora] = datos_laborables[hora]['Valor'].mean()
        medias_festivos[hora] = datos_festivos[hora]['Valor'].mean()
        medias_sabados[hora] = datos_sabados[hora]['Valor'].mean()
    
    fichero = open("normalizacion/medias_laborables.json", "w")
    json.dump(medias_laborables, fichero, indent=6)
    fichero.close()
    fichero = open("normalizacion/medias_festivos.json", "w")
    json.dump(medias_festivos, fichero, indent=6)
    fichero.close()
    fichero = open("normalizacion/medias_sabados.json", "w")
    json.dump(medias_sabados, fichero, indent=6)
    fichero.close()


def lee_datos_normalizacion(tipo):
    """Lee los datos de normalizacion previamente computados y guardados.

    Argumentos:
    tipo -- string que indica los datos a leer
    """
    ruta = "normalizacion/medias_{}.json".format(tipo)
    f = open(ruta)
    medias = json.load(f)
    f.close()
    return medias


def normaliza_sabado(dia, hora, medias_laborables, medias_festivos, medias_sabados):
    """Consigue el factor de conversion de una hora de un sabado para la
    normalizacion por aproximacion a dia laborable.

    Argumentos:
    dia -- dia sobre el que se quiere el factor
    hora -- hora sobre la que se quiere el factor
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    """
    rho = get_rho(dia, 'S')
    media_lab = medias_laborables[hora]
    media_fes = medias_festivos[hora]
    media_sab = medias_sabados[hora]
    div1 = media_lab / media_sab
    div2 = media_lab / media_fes
    ret = div1 * rho + div2 * (1 - rho)
    return ret


def normaliza_otro(dia, hora, dia_semana, medias_laborables, medias_festivos):
    """Consigue el factor de conversion de una hora de un dia distinto a
    sabado para la normalizacion por aproximacion a dia laborable.

    Argumentos:
    dia -- dia sobre el que se quiere el factor
    hora -- hora sobre la que se quiere el factor
    dia_semana -- dia de la semana del dia
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    """
    rho = get_rho(dia, dia_semana)
    media_lab = medias_laborables[hora]
    media_fes = medias_festivos[hora]
    div = media_lab / media_fes
    ret = div + rho * (1 - div)
    return ret


def normaliza(dia_semana, dia, hora, medias_laborables, medias_festivos, medias_sabados):
    """Consigue el factor de conversion de una hora de un dia cualquiera
    para la normalizacion por aproximacion a dia laborable.

    Argumentos:
    dia_semana -- dia de la semana del dia
    dia -- dia sobre el que se quiere el factor
    hora -- hora sobre la que se quiere el factor
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    """
    if dia_semana == 'S':
        return normaliza_sabado(dia, hora, medias_laborables, medias_festivos, medias_sabados)
    else:
        return normaliza_otro(dia, hora, dia_semana, medias_laborables, medias_festivos)


def indice_semana(dia_semana):
    """Devuelve el indice de un dia de la semana.
    
    Argumentos:
    dia_semana -- dia de la semana del que se quiere el indice
    """
    return semana.index(dia_semana)


def atras_7n_dias(n, diatxt):
    """Consigue el dia que fue 7*n dias atras (anyo-mes-dia).

    Argumentos:
    n -- cuantas semanas se quiere ir hacia atras
    diatxt -- dia sobre el que se quiere partir (anyo-mes-dia)
    """
    dia = int(diatxt[8:])
    mes = (int(diatxt[5:7]) + 10) % 12
    anyo = int(diatxt[:4])

    dia = dia - 7*n
    if dia < 1:
        dia = dia + meses[mes]
    else:
        mes = mes + 1
    
    mes = (mes + 1) % 12
    if mes == 0:
        mes = 12
        anyo = anyo - 1
    
    fecha = "{}-".format(anyo)
    if mes < 10:
        fecha = fecha + "0"
    fecha = fecha + "{}".format(mes) + "-"
    if dia < 10:
        fecha = fecha + "0"
    fecha = fecha + "{}".format(dia)

    return fecha


def estudio_temperatura(temperatura, valores, l_t, pot=1):
    """ Estudia la correlacion de los valores con la temperatura centrada
    por valor absoluto, y elevada a una potencia, en cada punto de una lista.

    Argumentos:
    temperatura -- serie con los valores de temperatura
    valores -- serie con los valores de demanda energetica
    l_t -- lista con los puntos sobre los que centrar
    pot -- potencia a elevar la temperatura centrada
    """
    l_corr = []
    val = valores[:n_est]
    tempe = temperatura[:n_est]
    cor = abs(val.corr(tempe))
    for t in l_t:
        aux = []
        for i in range(n_est):
            aux.append(abs(t-tempe[i])**pot)
        aux = pd.Series(aux)
        l_corr.append(abs(val.corr(aux)))
    return l_corr, cor