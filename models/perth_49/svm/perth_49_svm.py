#!/usr/bin/env python
# coding: utf-8

import logging
import time
import json
import sys
import os

from pathlib import Path


import pandas as pd
import numpy as np
import optuna

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('perth_49_svm_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


DATA_PATH = Path('./../../../data/')
CLEAN_FOLDER = DATA_PATH / 'clean/'
DATASET_FILE_NAME = 'perth_49.csv'
RANDOM_SEED = 8
N_TRIALS = 3000


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

def objective(trial):
    try:
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1, log = True)
        C = trial.suggest_float('C', 1e-2, 1e2, log = True)
        
        model = SVR(
            kernel = kernel,
            gamma = gamma,
            C = C
        ).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logging.info(f'Trial {trial.number}, parámetros: {trial.params}')
        logging.info(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}\n')
    
        trial.set_user_attr('mse', mse)
        trial.set_user_attr('mae', mae)
    
        return rmse
        
    except Exception as e:
        logging.warning(f'Error en el trial {trial.number}: {e}')
        
        return np.inf

if __name__ == '__main__':
    try:
        study = optuna.create_study(direction = 'minimize')
        
        start_total_trials_time = time.time()
        
        study.optimize(objective, n_trials = N_TRIALS)
        
        end_total_trials_time = time.time()
        
        total_trials_duration = end_total_trials_time - start_total_trials_time

        logging.info(f'El estudio con {N_TRIALS} trials se ejecutó en {total_trials_duration:.2f} segundos.')
        
        logging.info(f'Mejor RMSE encontrado: {study.best_value}')
        logging.info('Mejores hiperparámetros encontrados:')
        logging.info(study.best_params)
        
        df_trials = study.trials_dataframe()
        
        new_cols_names = {
            'number': 'trial_number',
            'value': 'rmse',
            'params_C': 'c',
            'params_gamma': 'gamma',
            'params_kernel': 'kernel',
            'user_attrs_mae': 'mae',
            'user_attrs_mse': 'mse'
        }
        
        df_final = df_trials.rename(columns = new_cols_names)
        df_final = df_final[['trial_number', 'state', 'kernel', 'gamma', 'c', 'mse', 'rmse', 'mae']]

        results_path = 'perth_49_svm_optuna_results.csv'
        df_final.to_csv(results_path, index = False)
        
        logging.info(f"Estudio de Optuna finalizado, archivo con los resultados creado en '{results_path}'.")
        
    except Exception as e:
        logging.critical(f'Ocurrió un error crítico durante el estudio de Optuna: {e}')
        sys.exit(1)

