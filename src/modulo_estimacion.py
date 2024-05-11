""" Modulo auxiliar de estimacion.

Aqui se implementa toda la funcionalidad relacionada con la hiperparametrizacion
de todos los estimadores, para Ridge Regression, Multilayer Perceptrons y SVMs, y
para todos los tipos de normalizacion. Tambien se implementa la funcionalidad para
medir los resultados.
"""

# Imports necesarios

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

from modulo_datos import meses, horas, semana, normaliza, festivos_nacionales, atras_7n_dias, n_est


# Variables de configuracion

n_score = 24 * 365
horas_atras = 12


# Funciones

def hiperparametriza_estimador(estimador, estimador_txt, param_grid, X, y, tipo):
    """Consigue un dataframe con los hiperparametros que minimizan el negative mean absolute error.
    Tambien guarda el estimador.

    Argumentos:
    estimador -- modelo de estimador a usar
    estimador_txt -- cadena que identifique al estimador
    param_grid -- diccionario con los valores de los hiperparametros a probar
    X -- datos de las 12 horas atras para estimar
    y -- datos de las 6 horas posteriores a estimar
    tipo -- informacion sobre la normalizacion y la temperatura usadas
    """
    regr = Pipeline([('scaler', StandardScaler()),
                     (estimador_txt, estimador)])
    y_transformer = StandardScaler()

    inner_estimator = TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)

    n_folds = 2
    kf = KFold(n_splits=n_folds)

    cv_estimator = GridSearchCV(estimator=inner_estimator,
                                param_grid=param_grid,
                                cv=kf,
                                scoring='neg_mean_absolute_error',
                                refit=True,
                                n_jobs=12,
                                return_train_score=True,
                                verbose=0)
    ret = cv_estimator.fit(X, y)

    ruta = "estimadores/{}/est_{}.pkl".format(estimador_txt, tipo)

    joblib.dump(ret, ruta)
    
    return ret


def info_mejores_estimadores(cv_estimator, param_grid):
    """Consigue un dataframe con los hiperparametros que minimizan el negative mean absolute error.
    
    Argumentos:
    cv_estimator -- estructura con la informacion de la hiperparametrizacion devuelta por hiperparametriza_estimador()
    param_grid -- diccionario con los valores de los hiperparametros que se probaron al hiperparametrizar
    """
    df_search = pd.DataFrame.from_dict(cv_estimator.cv_results_)
    claves_aux = param_grid.keys()
    claves = []
    for k in claves_aux:
        claves.append('param_' + k)
    claves.append('mean_test_score')
    ret = df_search.sort_values(by='mean_test_score', ascending=False)[claves]

    return ret


def n_dias_delante(dia_init, n):
    """Consigue el dia que fue 7*n dias atras (anyo-mes-dia).

    Argumentos:
    dia_init -- dia sobre el que se quiere partir (anyo-mes-dia)
    n -- cuantas semanas se quiere ir hacia atras
    """
    dia = int(dia_init[8:])
    mes = (int(dia_init[5:7]) + 11) % 12
    anyo = int(dia_init[:4])

    dia = dia + n
    while(dia > meses[mes]):
        dia = dia - meses[mes]
        mes = mes + 1
        if mes == 12:
            mes = 0
            anyo = anyo + 1
    
    mes = mes + 1
    
    fecha = "{}-".format(anyo)
    if mes < 10:
        fecha = fecha + "0"
    fecha = fecha + "{}".format(mes) + "-"
    if dia < 10:
        fecha = fecha + "0"
    fecha = fecha + "{}".format(dia)

    return fecha


def n_horas_delante(dia_semana_init, dia_init, hora_init, n):
    """Devuelve el dia y hora que es n horas adelante.
    
    Argumentos:
    dia_semana_init -- dia de la semana del dia de partida
    dia_init -- dia de partida (anyo-mes-dia)
    hora_init -- hora de partida
    n -- cantidad de horas hacia delante
    """
    horaId = horas.index(hora_init)
    horaIdFin = (horaId + n) % 24
    hora = horas[horaIdFin]

    nDias = (n + horaId) // 24
    dia = n_dias_delante(dia_init, nDias)

    diaSemanaId = semana.index(dia_semana_init)
    diaSemanaIdFin = (diaSemanaId + nDias) % 7
    diaSemana = semana[diaSemanaIdFin]

    return diaSemana, dia, hora


def desnormaliza1(valores, id_init, data, medias_laborables, medias_festivos, medias_sabados):
    """Desnormaliza los datos normalizados por aproximacion a dia laborable.
    
    Argumentos:
    valores -- array de numpy con los valores a desnormalizar
    id_init -- indice del primer valor
    data -- dataframe con todos los datos
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    """
    desnormalizados = []

    dia_semana_init = data['Dia_Semana'][id_init]
    dia_init = data['Dia'][id_init]
    hora_init = data['Hora'][id_init]

    for i in range(len(valores)):
        desnormalizados.append([])
        aux = [n_horas_delante(dia_semana_init, dia_init, hora_init, i+k) for k in range(len(valores[0]))]
        listaDiasSemana = [k[0] for k in aux]
        listaDias = [k[1] for k in aux]
        listaHoras = [k[2] for k in aux]
        for j in range(len(listaDiasSemana)):
            v = valores[i][j]
            dato = v / normaliza(listaDiasSemana[j], listaDias[j], listaHoras[j], medias_laborables, medias_festivos, medias_sabados)
            desnormalizados[i].append(dato)
    
    ret = np.array(desnormalizados)

    return ret


def desnormaliza2(valores, id_init, data):
    """Desnormaliza los datos normalizados por semana previa.
    
    Argumentos:
    valores -- array de numpy con los valores a desnormalizar
    id_init -- indice del primer valor
    data -- dataframe con todos los datos
    """
    desnormalizados = []

    dia_semana_init = data['Dia_Semana'][id_init]
    dia_init = data['Dia'][id_init]
    hora_init = data['Hora'][id_init]

    for i in range(len(valores)):
        desnormalizados.append([])
        aux = [n_horas_delante(dia_semana_init, dia_init, hora_init, i+k) for k in range(len(valores[0]))]
        listaDiasSemana = [k[0] for k in aux]
        listaDias = [k[1] for k in aux]
        for j in range(len(listaDiasSemana)):
            v = valores[i][j]
            dia_semana = listaDiasSemana[j]
            dia = listaDias[j]
            if dia in festivos_nacionales:
                n = (semana.index(dia_semana) + 1) % 7
                dato = v * data['Valor'][id_init+i+j-(24*n)]
            else:
                n = 1
                while atras_7n_dias(n, dia) in festivos_nacionales:
                    n = n + 1
                dato = v * data['Valor'][id_init+i+j-(24*7*n)]
            desnormalizados[i].append(dato)
    
    ret = np.array(desnormalizados)

    return ret


def desnormaliza3x(valores, id_init, x, data, medias_laborables, medias_festivos, medias_sabados):
    """Desnormaliza los datos normalizados mixtos x dias.
    
    Argumentos:
    valores -- array de numpy con los valores a desnormalizar
    id_init -- indice del primer valor
    x -- dias atras que se ha ido para el segundo paso de la normalizacion
    data -- dataframe con todos los datos
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    """
    desnormalizados = []

    for i in range(len(valores)):
        desnormalizados.append([])
        for j in range(len(valores[0])):
            v = valores[i][j]
            dato = v * data['Valor_Normalizado1'][id_init+i+j-(24*x)]
            desnormalizados[i].append(dato)
    
    aux = np.array(desnormalizados)

    ret = desnormaliza1(aux, id_init, data, medias_laborables, medias_festivos, medias_sabados)

    return ret


def score0(estimador, XTest, yTest, identidicador):
    """Consigue el score, en mean absolute error, de un estimador de datos sin normalizar.
    
    Argumentos:
    estimador -- estimador del que se quiere conocer el score
    XTest -- array de las variables regresoras
    yTest -- array de los valores reales
    identificador -- string que identifica el estimador
    """
    yPred = estimador.predict(XTest)
    aux = yPred.T.tolist()
    preds = pd.read_csv('preds.csv', sep=';', index_col=0)
    for i in range(len(aux)):
        clave = "{}_{}".format(identidicador, i)
        preds[clave] = aux[i]
    preds.to_csv(path_or_buf="preds.csv", sep=";", encoding="utf-8")
    ret = mean_absolute_error(yTest, yPred, multioutput='raw_values').tolist()
    return ret


def score1(estimador, XTest, yTest, data, medias_laborables, medias_festivos, medias_sabados, identidicador):
    """Consigue el score, en mean absolute error, de un estimador de datos con normalizacion por aproximacion a dia laborable.
    
    Argumentos:
    estimador -- estimador del que se quiere conocer el score
    XTest -- array de las variables regresoras
    yTest -- array de los valores reales
    data -- dataframe con todos los datos
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    identificador -- string que identifica el estimador
    """
    yPred = estimador.predict(XTest)
    yPredDesnor = desnormaliza1(yPred, n_est+horas_atras, data, medias_laborables, medias_festivos, medias_sabados)
    aux = yPredDesnor.T.tolist()
    preds = pd.read_csv('preds.csv', sep=';', index_col=0)
    for i in range(len(aux)):
        clave = "{}_{}".format(identidicador, i)
        preds[clave] = aux[i]
    preds.to_csv(path_or_buf="preds.csv", sep=";", encoding="utf-8")
    ret = mean_absolute_error(yTest, yPredDesnor, multioutput='raw_values').tolist()
    return ret


def score2(estimador, XTest, yTest, data, identidicador):
    """Consigue el score, en mean absolute error, de un estimador de datos con normalizacion por semana previa.
    
    Argumentos:
    estimador -- estimador del que se quiere conocer el score
    XTest -- array de las variables regresoras
    yTest -- array de los valores reales
    data -- dataframe con todos los datos
    identificador -- string que identifica el estimador
    """
    yPred = estimador.predict(XTest)
    yPredDesnor = desnormaliza2(yPred, n_est+horas_atras, data)
    aux = yPredDesnor.T.tolist()
    preds = pd.read_csv('preds.csv', sep=';', index_col=0)
    for i in range(len(aux)):
        clave = "{}_{}".format(identidicador, i)
        preds[clave] = aux[i]
    preds.to_csv(path_or_buf="preds.csv", sep=";", encoding="utf-8")
    ret = mean_absolute_error(yTest, yPredDesnor, multioutput='raw_values').tolist()
    return ret


def score3x(estimador, XTest, yTest, x, data, medias_laborables, medias_festivos, medias_sabados, identidicador):
    """Consigue el score, en mean absolute error, de un estimador de datos con normalizacion mixta x dias.
    
    Argumentos:
    estimador -- estimador del que se quiere conocer el score
    XTest -- array de las variables regresoras
    yTest -- array de los valores reales
    x -- dias atras que se ha ido para el segundo paso de la normalizacion
    data -- dataframe con todos los datos
    medias_laborables -- diccinario con las medias por hora de los dias laborables
    medias_festivos -- diccinario con las medias por hora de los dias festivos
    medias_sabados -- diccinario con las medias por hora de los sabados
    identificador -- string que identifica el estimador
    """
    yPred = estimador.predict(XTest)
    yPredDesnor = desnormaliza3x(yPred, n_est+horas_atras, x, data, medias_laborables, medias_festivos, medias_sabados)
    aux = yPredDesnor.T.tolist()
    preds = pd.read_csv('preds.csv', sep=';', index_col=0)
    for i in range(len(aux)):
        clave = "{}_{}".format(identidicador, i)
        preds[clave] = aux[i]
    preds.to_csv(path_or_buf="preds.csv", sep=";", encoding="utf-8")
    ret = mean_absolute_error(yTest, yPredDesnor, multioutput='raw_values').tolist()
    return ret