#!/usr/bin/env python
# coding: utf-8

import logging
import joblib
import copy
import time
import json
import sys
import os

from pathlib import Path


import pandas as pd
import random as rd
import numpy as np
import optuna


DATA_PATH = Path('./../../../data/')
CLEAN_FOLDER = DATA_PATH / 'clean/'
STUDY_NAME = 'perth_49'
DATASET_FILE_NAME = f'{STUDY_NAME}.csv'
CONFIGS_PATH = Path('./../configs/')
MODEL_FILE_NAME = f'{STUDY_NAME}_svm_best_model.joblib'
RANDOM_SEED = 8
N_TRIALS = 100


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(f'{STUDY_NAME}_pso_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


np.random.seed(RANDOM_SEED)
rd.seed(RANDOM_SEED)


try:
    df = pd.read_csv(CLEAN_FOLDER / DATASET_FILE_NAME)
    df_ = df.drop('Total_Power', axis = 1)
    feature_names = df_.columns.tolist()
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
    
try:
    model = joblib.load(CONFIGS_PATH / MODEL_FILE_NAME)
    logging.info(f"Modelo cargado desde '{CONFIGS_PATH / MODEL_FILE_NAME}'.")

except FileNotFoundError:
    logging.error(f"Error: no se encontró el archivo del modelo en '{CONFIGS_PATH / PARAMS_FILE_NAME}'.")
    sys.exit(1)
    
except Exception as e:
    logging.error(f"Ocurrió un error inesperado al cargar el modelo: {e}")
    sys.exit(1)


def f(X: np.array, feature_names: list = feature_names) -> float:
    n = X.shape[0]
    X_ = X.reshape((-1, n), copy = True)

    X_df = pd.DataFrame(X_, columns = feature_names)
    total_power = model.predict(X_df)

    return -total_power[0]


class Particle:

    def __init__(
        self,
        position: np.array,
        velocity: np.array,
        fitness: float,
        best_position: np.array,
        best_fitness: float
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.best_position = best_position
        self.best_fitness = best_fitness

    def __str__(self) -> str:
        pos = f'Posición: {self.position}'
        vel = f'Velocidad: {self.velocity}'
        fit = f'Aptitud: {self.fitness}'
        best_pos = f'Mejor posición: {self.best_position}'
        best_fit = f'Mejor aptitud: {self.best_fitness}'

        return f'{pos}\n{vel}\n{fit}\n{best_pos}\n{best_fit}'
    
    def update_velocity(
        self,
        W: float,
        C1: float,
        C2: float,
        best_global_position: np.array
    ) -> None:
        R1 = rd.random()
        R2 = rd.random()

        self.velocity = W*self.velocity + C1*R1*(self.best_position - self.position) + C2*R2*(best_global_position - self.position)

    def update_position(
        self,
        low: np.array,
        high: np.array,
        use_box_constraints: bool
    ) -> None:
        self.position += self.velocity

        if use_box_constraints:
            self.position = np.clip(self.position, low, high)

    def update_fitness(
        self,
        f: callable
    ) -> None:
        self.fitness = f(self.position)

        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
    
    def update_particle(
        self,
        f: callable,
        low: np.array,
        high: np.array,
        W: float,
        C1: float,
        C2: float,
        best_global_position: np.array,
        use_box_constraints: bool
    ) -> None:
        self.update_velocity(W, C1, C2, best_global_position)
        self.update_position(low, high, use_box_constraints)
        self.update_fitness(f)


def get_init_pop(
    f: callable, 
    n: int,
    size: int,
    low: np.array,
    high: np.array
) -> list[Particle]:
    pop = []

    for _ in range(size):
        position = np.array([rd.uniform(low[i], high[i]) for i in range(n)])
        velocity = np.zeros(n)
        fitness = f(position)
        best_position = position.copy()
        best_fitness = fitness

        pop.append(Particle(position, velocity, fitness, best_position, best_fitness))

    return pop


def find_best_global_particle(pop: list[Particle]) -> Particle:
    best_particle = min(pop, key = lambda p: p.best_fitness)

    return copy.deepcopy(best_particle)


def ParticleSwarmOptimization(
    f: callable,
    n: int,
    low: np.array,
    high: np.array,
    size: int,
    W: float,
    C1: float,
    C2: float,
    max_iter: int,
    use_box_constraints: bool = True
) -> Particle:
    """
    Ejecuta el algoritmo de Optimización por Enjambre de Partículas (PSO).

    Este algoritmo minimiza una función objetivo moviendo un enjambre de
    partículas a través del espacio de búsqueda.

    Args:
        f (callable): la función objetivo a minimizar. Debe aceptar un np.array y devolver un float.
        n (int): la dimensionalidad del problema (número de variables).
        low (np.array): el límite inferior de las variables de búsqueda.
        high (np.array): el límite superior de las variables de búsqueda.
        size (int): el número de partículas en el enjambre.
        W (float): el coeficiente de inercia. Controla la influencia de la velocidad anterior.
        C1 (float): el coeficiente cognitivo. Atrae la partícula a su mejor posición histórica.
        C2 (float): el coeficiente social. Atrae la partícula a la mejor posición global del enjambre.
        max_iter (int): el número máximo de iteraciones.
        use_box_constraints (bool, optional): si se deben aplicar restricciones de caja. Defaults to True.

    Returns:
        Particle: la partícula con la mejor posición y aptitud encontrada en todo el enjambre.
    """
    pop = get_init_pop(f, n, size, low, high)
    best_global = find_best_global_particle(pop)

    for _ in range(max_iter):
        for particle in pop:
            particle.update_particle(f, low, high, W, C1, C2, best_global.position, use_box_constraints)
        
        best_global_candidate = find_best_global_particle(pop)

        if best_global_candidate.best_fitness < best_global.best_fitness:
            best_global = copy.deepcopy(best_global_candidate)

    return best_global


try:
    num_of_wecs = int(DATASET_FILE_NAME.split('_')[1].split('.')[0])
    
    logging.info('Problema de optimización inicializado.')
    logging.info(f'Número de intentos para el algoritmo de PSO: {N_TRIALS}')

except IndexError:
    logging.error(f"Error: el nombre del archivo '{DATASET_FILE_NAME}' no tiene el formato esperado.")
    sys.exit(1)

except Exception as e:
    logging.error(f"Ocurrió un error inesperado al inicializar el problema: {e}")
    sys.exit(1)


def objective(trial):
    try:
        n = 2 * num_of_wecs
        
        size = trial.suggest_int('size', 10, 300, log = True)
        W = trial.suggest_float('W', 0.1, 0.9)
        C1 = trial.suggest_float('C1', 1.5, 2.5)
        C2 = trial.suggest_float('C2', 1.5, 2.5)
        max_iter = trial.suggest_int('max_iter', 10, 1000, log = True)
    
        res = ParticleSwarmOptimization(
            f = f,
            n = n,
            low = np.zeros(n),
            high = np.ones(n),
            size = size,
            W = W,
            C1 = C1,
            C2 = C2,
            max_iter = max_iter
        )

        best_positions = res.best_position
        best_total_power = res.best_fitness

        if best_total_power > -np.inf:
        
            trial.set_user_attr('best_positions', best_positions)
            trial.set_user_attr('best_total_power', best_total_power)
    
            logging.info(f'Trial {trial.number}.')
            logging.info(f'Mejor valor encontrado hasta el momento: {-best_total_power}')
            
            return best_total_power
    
        else:
            logging.info(f'Trial {trial.number} no encontró una solución válida.')
            
            return np.inf
            
    except Exception as e:
        logging.warning(f'Error en el trial {trial.number}: {e}')
        
        return np.inf

if __name__ == '__main__':
    try:
        logging.info(f'Iniciando estudio de Optuna para el algoritmo de PSO.')
        
        study = optuna.create_study(
            sampler = optuna.samplers.TPESampler(seed = RANDOM_SEED),
            direction = 'minimize'
        )
    
        start_total_trials_time = time.time()
        
        study.optimize(objective, n_trials = N_TRIALS)
        
        end_total_trials_time = time.time()
        
        total_trials_duration = end_total_trials_time - start_total_trials_time
    
        logging.info(f'Estudio de Optuna para el algoritmo de PSO finalizado en {total_trials_duration:.2f} segundos.')
    
        logging.info(f'Mejor valor encontrado: {-study.best_value}')
        logging.info(f'Mejores hiperparámetros encontrados:')
        
        best_params = study.best_params 
        logging.info(best_params)
    
        df_trials = study.trials_dataframe().copy()
    
        df_trials.columns = df_trials.columns.str.replace('params_', '')
        df_trials.columns = df_trials.columns.str.replace('user_attrs_', '')
        
        new_cols_names = {
            'number': 'trial_number'
        }
        
        df_final = df_trials.rename(columns = new_cols_names)
        
        cols_to_drop = [
            'value',
            'datetime_start',
            'datetime_complete',
            'duration'   
        ]
        
        df_final = df_final.drop(columns = cols_to_drop)
        
        results_path = f'{STUDY_NAME}_pso_optuna_results.csv'
        df_final.to_csv(results_path, index = False)
                
        logging.info(f"Archivo de resultados para el algoritmo de PSO creado en '{results_path}'.")
        logging.info(f"Guardando los mejores hiperparámetros para el algoritmo de PSO.")

        hyperparams_file_path = f'{STUDY_NAME}_pso_best_hyperparams.json'
        with open(hyperparams_file_path, "w") as fp:
            json.dump(best_params, ffp, indent = 4)

        logging.info(f"Archivo con los mejores hiperparámetros para el algoritmo de PSO creado en '{hyperparams_file_path}'.")
        
    except Exception as e:
        logging.critical(f'Ocurrió un error crítico durante el estudio de Optuna: {e}')
        sys.exit(1)

