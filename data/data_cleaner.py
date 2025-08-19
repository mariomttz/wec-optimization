#!/usr/bin/env python
# coding: utf-8

import logging
import time
import json
import sys
import os

from pathlib import Path


import pandas as pd


logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('data_cleaner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


RAW_FOLDER = Path('raw/')
CLEAN_FOLDER = Path('clean/')


def min_max_normalization(col): return (col - col.min()) / (col.max() - col.min())


def z_score(col): return (col - col.mean()) / col.std()


try:
    logging.info(f"Buscando archivos CSV en el directorio '{RAW_FOLDER}'.")
    csv_files = list(RAW_FOLDER.rglob('*.csv'))
    
    if not csv_files:
        logging.warning(f"No se encontraron archivos CSV en el directorio '{RAW_FOLDER}'.")
        sys.exit(0)

except Exception as e:
    logging.critical(f"Error crítico al acceder al directorio '{RAW_FOLDER}': {e}")
    sys.exit(1)

logging.info(f"Procesando los archivos CSV encontrados en el directorio '{RAW_FOLDER}'.")
for file in csv_files:
    
    try:
        file_name = file.stem
        parts = file_name.split('_')
        
        if len(parts) < 3:
            logging.error(f"Nombre del archivo inválido: '{file_name}'.")
            continue
        
        city = parts[1]
        
        try:
            num_of_wecs = int(parts[2])
            
        except (ValueError, IndexError):
            logging.error(f"El número de WECs no es válido en el nombre del archivo: '{file_name}'.")
            continue
            
        logging.info(f"Procesando el archivo: {file_name}")
        logging.info(f"Ruta del archivo: {file}")
        logging.info(f"Ciudad: {city}")
        logging.info(f"Número de WECs: {num_of_wecs}")
        
        try:
            df = pd.read_csv(file)
            logging.info('Datos cargados correctamente.')
        
        except FileNotFoundError:
            logging.error(f"Error: no se encontró el archivo en '{file}'.")
            sys.exit(1)
            
        except pd.errors.EmptyDataError:
            logging.error(f"Error: el archivo '{file}' está vacío.")
            sys.exit(1)
            
        except Exception as e:
            logging.error(f"Ocurrió un error inesperado al leer el archivo '{file}': {e}")
            sys.exit(1)

        original_rows = df.shape[0]
        missing_values = df.isnull().sum().sum()
        duplicate_rows_count = df.duplicated().sum()

        logging.info(f"Número de registros antes del procesamiento: {original_rows}")
        logging.info(f"Número de registros faltantes: {missing_values}")
        logging.info(f"Número de registros duplicados: {duplicate_rows_count}")
        
        df_copy = df.copy()
        cols_to_delete = df_copy.columns[num_of_wecs * 2:-1]
        
        if not cols_to_delete.empty:
            df_copy.drop(columns = cols_to_delete, axis = 1, inplace = True)
            logging.info(f"Columnas eliminadas: {list(cols_to_delete)}")
            
        else:
            logging.info('No se eliminaron columnas.')

        df_copy.drop_duplicates(inplace = True)
        cleaned_rows = df_copy.shape[0]
        logging.info(f"Número de registros después del procesamiento: {cleaned_rows}")

        if cleaned_rows == 0:
            logging.warning(f"El conjunto '{file_name}' está vacío después de la limpieza. Saltando el procesamiento.")
            sys.exit(1)

        logging.info('Obteniendo estadísticos del conjunto antes de la normalización y estandarización.')
        cols_to_modify = df_copy.columns[:-1]
        mins = df_copy[cols_to_modify].min().to_dict()
        maxs = df_copy[cols_to_modify].max().to_dict()
        total_power_mean = df_copy['Total_Power'].mean()
        total_power_std = df_copy['Total_Power'].std()
        
        raw_stats = {
            'mins': mins,
            'maxs': maxs,
            'total_power_mean': total_power_mean,
            'total_power_std': total_power_std
        }
        
        new_file_name = f'{city.lower()}_{num_of_wecs}'
        
        try:
            CLEAN_FOLDER.mkdir(parents = True, exist_ok = True)
            
            with open(CLEAN_FOLDER / (new_file_name + '_raw_stats' + '.json'), 'w') as fp:
                json.dump(raw_stats, fp, indent = 4)
            logging.info(f"Estadísticas guardadas en '{CLEAN_FOLDER / new_file_name}_raw_stats.json'.")

            logging.info('Normalización y estandarizando el conjunto de datos.')
            df_copy[cols_to_modify] = min_max_normalization(df_copy[cols_to_modify])
            df_copy['Total_Power'] = z_score(df_copy['Total_Power'])
            
            df_copy.to_csv(CLEAN_FOLDER / (new_file_name + '.csv'), index = False)
            logging.info(f"Conjunto de datos limpio guardado en '{CLEAN_FOLDER / new_file_name}.csv'.")

        except Exception as e:
            logging.error(f"Error al escribir los archivos para '{new_file_name}': {e}")
            sys.exit(1)
            
    except Exception as e:
        logging.critical(f"Ocurrió un error crítico inesperado al procesar '{file_name}': {e}")
        sys.exit(1)

