#!/usr/bin/env python
# coding: utf-8

import logging
import joblib
import time
import json
import sys
import os

from pathlib import Path


import pandas as pd
import random as rd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


DATA_PATH = Path('./../../../data/')
CLEAN_FOLDER = DATA_PATH / 'clean/'
STUDY_NAME = 'sydney_49'
DATASET_FILE_NAME = f'{STUDY_NAME}.csv'
RANDOM_SEED = 8


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(f'{STUDY_NAME}_lr.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


np.random.seed(RANDOM_SEED)
rd.seed(RANDOM_SEED)


try:
    df = pd.read_csv(CLEAN_FOLDER / DATASET_FILE_NAME)
    logging.info('Datos cargados correctamente.')
    
except FileNotFoundError:
    logging.error(f"Error: no se encontró el archivo en '{CLEAN_FOLDER / DATASET_FILE_NAME}'.")
    sys.exit(1)
    
except pd.errors.EmptyDataError:
    logging.error(f"Error: el archivo '{DATASET_FILE_NAME}' está vacío.")
    sys.exit(1)
    
except Exception as e:
    logging.error(f"Ocurrió un error inesperado al leer el archivo '{DATASET_FILE_NAME}': {e}")
    sys.exit(1)

X = df.drop('Total_Power', axis = 1)
y = df['Total_Power']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.2, 
    random_state = RANDOM_SEED
)

try:
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    coeficientes = model.coef_
    intercepto = model.intercept_
    
    logging.info(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
    logging.info(f'Coeficientes: {coeficientes}')
    logging.info(f'Intercepto: {intercepto}')

except Exception as e:
    logging.critical(f'Ocurrió un error crítico durante el entrenamiento o la evaluación del modelo: {e}')
    sys.exit(1)

try:
    results_data = {
        'mse': [mse],
        'rmse': [rmse],
        'mae': [mae],
        'intercepto': [intercepto]
    }
    
    for i, feature in enumerate(X.columns):
        results_data[f'coef_{feature}'] = [coeficientes[i]]

    df_final = pd.DataFrame(results_data)
    
    results_path = f'{STUDY_NAME}_lr_results.csv'
    df_final.to_csv(results_path, index = False)
    
    logging.info(f"Archivo con los resultados creado en '{results_path}'.")
    logging.info('Entrenando el modelo.')

    model_path = f'{STUDY_NAME}_lr_best_model.joblib'
    joblib.dump(model, model_path)

    logging.info(f"Modelo entrenado guardado en '{model_path}'")

except Exception as e:
    logging.error(f'Ocurrió un error al guardar los resultados: {e}')
    sys.exit(1)

