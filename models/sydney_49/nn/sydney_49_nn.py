#!/usr/bin/env python
# coding: utf-8

import logging
import time
import json
import sys
import os

from pathlib import Path


import pandas as pd
import random as rd
import numpy as np
import optuna

import fastai.metrics as fm
from fastai.tabular.all import *

import torch.nn.functional as F
import torch.nn as nn

from torch.nn import SmoothL1Loss


DATA_PATH = Path('./../../../data/')
CLEAN_FOLDER = DATA_PATH / 'clean/'
STUDY_NAME = 'sydney_49'
DATASET_FILE_NAME = f'{STUDY_NAME}.csv'
RANDOM_SEED = 8
BATCH_SIZE = 1024
N_TRIALS = 1000


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(f'{STUDY_NAME}_nn_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


np.random.seed(RANDOM_SEED)
rd.seed(RANDOM_SEED)


class ResBlock(nn.Module):
    def __init__(self, n_fts):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(n_fts),
            nn.Linear(n_fts, n_fts//2),
            nn.ReLU(),
            nn.BatchNorm1d(n_fts//2),
            nn.Linear(n_fts//2, n_fts),
            nn.BatchNorm1d(n_fts)
        )
        self.activation = nn.SELU()
        
    def forward(self, x):
        return self.activation(
            x + self.block(x)
        )


class PowerEstimator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim = 1,
        use_dropout = True,
        dropout_rate = 0.5,
        use_batchnorm = True,
        use_resblocks = True,
        num_resblocks = 1
    ):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.SELU())

            if use_dropout:
                layers.append(nn.Dropout1d(p = dropout_rate))

            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.resblocks = nn.ModuleList()
        
        if use_resblocks:
            self.resblocks = nn.ModuleList([
                ResBlock(current_dim) for _ in range(num_resblocks)
            ])
        
        self.output_layer = nn.Linear(current_dim, output_dim)
        
    def forward(self, _, x):
        x = self.hidden_layers(x)

        for res_block in self.resblocks:
            x = res_block(x)

        return self.output_layer(x)


def load_data(df, cont_names, y_names, batch_size = BATCH_SIZE):
    to = TabularPandas(
        df = df,
        cont_names = cont_names,
        y_names = y_names,
        y_block = RegressionBlock,
        splits = RandomSplitter(
            valid_pct = 0.2,
            seed = RANDOM_SEED
        )(df)
    )
    
    return to.dataloaders(bs = batch_size)


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

cont_names = list(df.columns[:-1])
y_names = 'Total_Power'

dls = load_data(df, cont_names, y_names)
input_dim = dls.train.xs.shape[1]

def objective(trial):
    try:
        num_layers = trial.suggest_int('num_layers', 1, 5)
        
        hidden_dims = [
            trial.suggest_int(f'num_units_layer_{i}', 16, 256, log = True)
            for i in range(num_layers)
        ]

        use_dropout = trial.suggest_categorical('use_dropout', [True, False])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.8)
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        use_resblocks = trial.suggest_categorical('use_resblocks', [True, False])
        num_resblocks = 0

        if use_resblocks:
            num_resblocks = trial.suggest_int('num_resblocks', 1, 5)

        n_epoch = trial.suggest_int('n_epoch', 10, 300, log = True)
        lr_max = trial.suggest_float('lr_max', 1e-6, 1e-2, log = True)

        model = PowerEstimator(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            use_dropout = use_dropout,
            dropout_rate = dropout_rate,
            use_batchnorm = use_batchnorm,
            use_resblocks = use_resblocks,
            num_resblocks = num_resblocks
        )

        learn = Learner(
            dls,
            model = model,
            loss_func = SmoothL1Loss(),
            opt_func = Adam,
            metrics = [
                F.mse_loss,
                fm.rmse,
                F.l1_loss
            ]
        )

        learn.fit_one_cycle(
            n_epoch = n_epoch,
            lr_max = lr_max,
            div = 25.0,
            div_final = 100000.0,
            pct_start= 0.25
        )

        mse = learn.recorder.final_record[2]
        rmse = learn.recorder.final_record[3]
        mae = learn.recorder.final_record[4]

        logging.info(f'Trial {trial.number}, parámetros: {trial.params}')
        logging.info(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

        trial.set_user_attr('mse', mse)
        trial.set_user_attr('rmse', rmse)
        trial.set_user_attr('mae', mae)

        return learn.recorder.final_record[1]

    except Exception as e:
        logging.warning(f'Error en el trial {trial.number}: {e}')
        
        return np.inf
        
if __name__ == '__main__':
    try:
        study = optuna.create_study(
            sampler = optuna.samplers.TPESampler(seed = RANDOM_SEED),
            direction = 'minimize'
        )
        
        start_total_trials_time = time.time()
        
        study.optimize(objective, n_trials = N_TRIALS)
        
        end_total_trials_time = time.time()
        
        total_trials_duration = end_total_trials_time - start_total_trials_time

        logging.info(f'El estudio con {N_TRIALS} trials se ejecutó en {total_trials_duration:.2f} segundos.')
        
        logging.info(f'Mejor L1 Smooth: {study.best_value}')
        logging.info('Mejores hiperparámetros encontrados:')
        logging.info(study.best_params)

        df_trials = study.trials_dataframe().copy()

        df_trials.columns = df_trials.columns.str.replace('params_', '')
        df_trials.columns = df_trials.columns.str.replace('user_attrs_', '')

        new_cols_names = {
            'number': 'trial_number',
            'value': 'l1_smooth',
        }

        df_final = df_trials.rename(columns = new_cols_names)

        cols_to_drop = [
            'datetime_start',
            'datetime_complete',
            'duration'   
        ]

        df_final = df_final.drop(columns = cols_to_drop)

        results_path = f'{STUDY_NAME}_nn_optuna_results.csv'
        df_final.to_csv(results_path, index = False)
        
        logging.info(f"Estudio de Optuna finalizado, archivo con los resultados creado en '{results_path}'.")
        logging.info('Entrenando el modelo con los mejores hiperparámetros encontrados.')
        
        best_params = study.best_params
        
        best_model = PowerEstimator(
            input_dim = input_dim,
            hidden_dims = [best_params[f'num_units_layer_{i}'] for i in range(best_params['num_layers'])],
            use_dropout = best_params['use_dropout'],
            dropout_rate = best_params['dropout_rate'],
            use_batchnorm = best_params['use_batchnorm'],
            use_resblocks = best_params['use_resblocks'],
            num_resblocks = best_params.get('num_resblocks', 0)
        )
        
        learn = Learner(
            dls,
            model = best_model,
            loss_func = SmoothL1Loss(),
            opt_func = Adam,
            metrics = [
                F.mse_loss,
                fm.rmse,
                F.l1_loss
            ]
        )
        
        learn.fit_one_cycle(
            n_epoch = best_params['n_epoch'],
            lr_max = best_params['lr_max'],
            div = 25.0,
            div_final = 100000.0,
            pct_start= 0.25
        )

        hyperparams_path = f'{STUDY_NAME}_nn_hyperparams_best_model.json'
        
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params, f)
            
        logging.info(f"Hiperparámetros del mejor modelo guardados en '{hyperparams_path}'.")
        
        model_path = f'{STUDY_NAME}_nn_best_model.pth'
        torch.save(learn.model.state_dict(), model_path)
        
        logging.info(f"Modelo entrenado con los mejores hiperparámetros encontrados guardado en '{model_path}'.")

    except Exception as e:
        logging.critical(f'Ocurrió un error crítico durante el estudio de Optuna: {e}')
        sys.exit(1)

