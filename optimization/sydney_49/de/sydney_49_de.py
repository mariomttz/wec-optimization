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

from catboost import CatBoostRegressor


DATA_PATH = Path('./../../../data/')
CLEAN_FOLDER = DATA_PATH / 'clean/'
STUDY_NAME = 'sydney_49'
DATASET_FILE_NAME = f'{STUDY_NAME}.csv'
CONFIGS_PATH = Path('./../configs/')
MODEL_FILE_NAME = f'{STUDY_NAME}_catboost_best_model.cbm'
RANDOM_SEED = 8
N_TRIALS = 100


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(f'{STUDY_NAME}_de_optimization.log'),
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
    model = CatBoostRegressor()
    model.load_model(CONFIGS_PATH / MODEL_FILE_NAME)
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


def get_init_pop(
    f: callable,
    n: int,
    size: int,
    low: np.array,
    high: np.array
) -> list:
    pop = []

    for _ in range(size):
        x = np.random.uniform(low, high, size = n)
        pop.append((x, f(x)))

    return pop


def m_rand(
    pop: list,
    F: float,
    low: np.array,
    high: np.array,
    use_box_constraints: bool
) -> np.array:
    r0, r1, r2 = rd.sample(population = pop, k = 3)
    v = r0[0] + F*(r1[0] - r2[0])

    if use_box_constraints:
        return np.clip(v, low, high, out = v)

    return v

def m_best(
    pop:list,
    F: float,
    low: np.array,
    high: np.array,
    use_box_constraints: bool
) -> np.array:
    pop.sort(key = lambda x: x[1])
    
    best = pop[0]
    r1, r2 = rd.sample(population = pop, k = 2)
    v = best[0] + F*(r1[0] - r2[0])

    if use_box_constraints:
        return np.clip(v, low, high, out = v)

    return v


def c_bin(
    x: np.array,
    v: np.array,
    CR: float,
) -> np.array:
    u = []
    j_rand = np.random.choice(list(range(len(x))))

    for j in range(len(x)):
        if np.random.uniform(low = 0, high = 1) < CR or j == j_rand:
            u.append(v[j])
        else:
            u.append(x[j])

    return np.array(u)

def c_exp(
    x: np.array,
    v: np.array,
    CR: float
) -> np.array:
    u = np.zeros_like(x)
    n = x.shape[0]
    i_rand = rd.randint(0, n)
    L_rand = rd.randint(1, n)

    for i in range(n):
        k = (i_rand + i) % n
        
        if np.random.uniform(low = 0, high = 1) <= CR or i == 0:
            u[k] = v[k]
        else:
            break

    for i in range(n):
        if u[i] == 0:
            u[i] = x[i]
            
    return u


def DifferentialEvolution(
    f: callable,
    n: int,
    low: np.array,
    high: np.array,
    F: float,
    CR: float,
    size: int,
    G: int,
    f_mutation: str = 'rand',
    f_crossover: str = 'bin',
    use_box_constraints: bool = True
) -> tuple:
    """
    Ejecuta el algoritmo de Evolución Diferencial para minimizar una función objetivo.

    Args:
        f (callable): la función objetivo a minimizar. Debe aceptar un np.array y devolver un float.
        n (int): la dimensionalidad del problema (número de variables).
        low (np.array): el límite inferior de las variables de búsqueda.
        high (np.array): el límite superior de las variables de búsqueda.
        F (float): el factor de mutación, generalmente en el rango [0, 2].
        CR (float): la tasa de cruce, en el rango [0, 1].
        size (int): el tamaño de la población.
        G (int): el número de generaciones (iteraciones).
        f_mutation (str, optional): estrategia de mutación ('rand' o 'best'). Defaults to 'rand'.
        f_crossover (str, optional): estrategia de cruce ('bin' o 'exp'). Defaults to 'bin'.
        use_box_constraints (bool, optional): si se deben aplicar restricciones de caja. Defaults to True.

    Returns:
        tuple: una tupla con (best_solution_vector, best_fitness_value).
    """
    if f_mutation == 'rand':
        mutation = m_rand
    
    else:
        mutation = m_best

    if f_crossover == 'bin':
        crossover = c_bin

    else:
        crossover = c_exp
        
    pop = get_init_pop(f, n, size, low, high)

    for _ in range(G):
        new_pop = []

        for x in pop:
            v = mutation(pop, F, low, high, use_box_constraints)
            u = crossover(x[0], v, CR)
            f_new = f(u)

            if f_new < x[1]:
                new_pop.append((u, f_new))
            else:
                new_pop.append(x)

        pop = new_pop

    best = min(pop, key = lambda x: x[1])

    return best


try:
    num_of_wecs = int(DATASET_FILE_NAME.split('_')[1].split('.')[0])
    
    logging.info('Problema de optimización inicializado.')
    logging.info(f'Número de intentos por variante: {N_TRIALS}')

except IndexError:
    logging.error(f"Error: el nombre del archivo '{DATASET_FILE_NAME}' no tiene el formato esperado.")
    sys.exit(1)

except Exception as e:
    logging.error(f"Ocurrió un error inesperado al inicializar el problema: {e}")
    sys.exit(1)


def objective(trial, variant):
    try:
        n = 2 * num_of_wecs
        f_mutation = variant.split('/')[1]
        f_crossover = variant.split('/')[3]
        
        F = trial.suggest_float('F', 0.1, 1.0)
        CR = trial.suggest_float('CR', 0.0, 1.0)
        size = trial.suggest_int('size', 10, 300, log = True)
        G = trial.suggest_int('G', 10, 1000, log = True)
    
        res = DifferentialEvolution(
            f = f,
            n = n,
            low = np.zeros(n),
            high = np.ones(n),
            F = F,
            CR = CR,
            size = size,
            G = G,
            f_mutation = f_mutation,
            f_crossover = f_crossover
        )

        best_positions = res[0]
        best_total_power = res[1]

        if best_total_power > -np.inf:
        
            trial.set_user_attr('best_positions', best_positions)
            trial.set_user_attr('best_total_power', best_total_power)
    
            logging.info(f'Trial {trial.number} de la variante {variant}.')
            logging.info(f'Mejor valor encontrado hasta el momento: {-best_total_power}')
            
            return best_total_power
    
        else:
            logging.info(f'Trial {trial.number} de la variante {variant} no encontró una solución válida.')
            
            return np.inf
            
    except Exception as e:
        logging.warning(f'Error en el trial {trial.number} ({variant}): {e}')
        
        return np.inf

if __name__ == '__main__':
    de_variants = [
        "DE/rand/1/bin",
        "DE/rand/1/exp",
        "DE/best/1/bin",
        "DE/best/1/exp",
    ]

    try:
        for variant in de_variants:
            logging.info(f'Iniciando estudio de Optuna para la variante: {variant}.')
            
            study = optuna.create_study(
                sampler = optuna.samplers.TPESampler(seed = RANDOM_SEED),
                direction = 'minimize'
            )
        
            start_total_trials_time = time.time()
            
            study.optimize(
                lambda trial: objective(trial, variant),
                n_trials = N_TRIALS
            )
            
            end_total_trials_time = time.time()
            
            total_trials_duration = end_total_trials_time - start_total_trials_time
        
            logging.info(f'Estudio de Optuna para la variante {variant} finalizado en {total_trials_duration:.2f} segundos.')
        
            logging.info(f'Mejor valor encontrado para la variante {variant}: {-study.best_value}')
            logging.info(f'Mejores hiperparámetros encontrados para la variante {variant}:')
            
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
            
            results_path = f'{STUDY_NAME}_de_{variant.replace("/", "-")}_optuna_results.csv'
            df_final.to_csv(results_path, index = False)
                    
            logging.info(f"Archivo de resultados para la variante {variant} creado en '{results_path}'.")
            logging.info(f"Guardando los mejores hiperparámetros para la variante {variant}.")

            variant_file_path = f'{STUDY_NAME}_de_{variant.replace("/", "-")}_best_hyperparams.json'
            with open(variant_file_path, "w") as fp:
                json.dump(best_params, fp, indent = 4)

            logging.info(f"Archivo con los mejores hiperparámetros para la variante {variant} creado en '{variant_file_path}'.")
        
    except Exception as e:
        logging.critical(f'Ocurrió un error crítico durante el estudio de Optuna: {e}')
        sys.exit(1)

