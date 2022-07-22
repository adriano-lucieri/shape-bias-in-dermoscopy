import os
import sys
import joblib
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import classification_report

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def make_dir_if_not_exists(path):
    """ Create folderstructure if not existent """
    if not os.path.exists(path):
        os.makedirs(path)

def save_evaluation_results(arr_predictions, arr_labels, output_dir, target_names=None):
    joblib.dump(arr_predictions, os.path.join(output_dir, 'testing_predictions.pkl'))
    joblib.dump(arr_labels, os.path.join(output_dir, 'testing_labels.pkl'))

    dct_report = classification_report(arr_labels, arr_predictions, target_names=target_names, zero_division=1, output_dict=True)
    joblib.dump(dct_report, os.path.join(output_dir, 'test_results_overall.pkl'))

def average_n_run_results_eval(target_folder, target_dataset, num_runs, run_comment='random'):

    def process_files(lst_files):
        ### Helper function which processes the target result pickle files. ###
        lst_dcts = list()

        for file in lst_files: 
            lst_dcts.append(joblib.load(file))
        num_runs = len(lst_dcts)

        column_names = ['class', 'precision-mean', 'precision-std', 'recall-mean', 'recall-std', 'f1-score-mean', 'f1-score-std', 'support', 'runs']
        df = pd.DataFrame(columns = column_names)
        
        lst_classes = list(lst_dcts[0].keys())
        lst_metrics = list(lst_dcts[0][lst_classes[0]].keys())

        for idx, c in enumerate(lst_classes):
            df.loc[idx] = '0' # Initialize row
            df.loc[idx]['class'] = c

            if c != 'accuracy':
                for m in lst_metrics:
                    if m not in ['support']:
                        lst_values = list()
                        for d in lst_dcts:
                            lst_values.append(d[c][m])

                        df.loc[idx][m+'-mean'] = np.mean(lst_values)
                        df.loc[idx][m+'-std'] = np.std(lst_values)
                    else:
                        df.loc[idx][m] = d[c][m]
            else:
                lst_values = list()
                for d in lst_dcts:
                    lst_values.append(d[c])
                df.loc[idx]['precision-mean'] = np.mean(lst_values)
                df.loc[idx]['precision-std'] = np.std(lst_values)
                df.loc[idx]['runs'] = num_runs
                
        for c in column_names[1:]:
            df[c] = pd.to_numeric(df[c])

        return df

    # Gather all target result pkl files
    dct_target_results = defaultdict(lambda: defaultdict(defaultdict))

    for run_id in range(num_runs):

        run_folder = f'{target_folder}/{run_comment}_run-{run_id}/{target_dataset}/'
        for path in Path(run_folder).rglob('test_results_overall.pkl'):
            filename = path.name

            run = str(path.parent).split('/')[-2]
            dct_target_results[run][filename] = path
        
    # Generate all averaged files
    lst_runs = list(dct_target_results.keys())
    lst_filetypes = list(dct_target_results[lst_runs[0]].keys())

    # Process the target result files
    for filetype in lst_filetypes:
        lst_rel_files = []
        for run in lst_runs:
            lst_rel_files.append(dct_target_results[run][filetype])
        df = process_files(lst_rel_files)

        target_filename = os.path.join(target_folder, f'avg_{target_dataset}_' + lst_rel_files[0].stem)
        df.to_csv(target_filename + '.csv', float_format='%.4f')
        joblib.dump(df, target_filename + '.pkl')

def average_n_run_results(target_folder):

    def process_files(lst_files):
        ### Helper function which processes the target result pickle files. ###
        lst_dcts = list()

        for file in lst_files: 
            lst_dcts.append(joblib.load(file))
        num_runs = len(lst_dcts)

        column_names = ['class', 'precision-mean', 'precision-std', 'recall-mean', 'recall-std', 'f1-score-mean', 'f1-score-std', 'support', 'runs']
        df = pd.DataFrame(columns = column_names)
        
        lst_classes = list(lst_dcts[0].keys())
        lst_metrics = list(lst_dcts[0][lst_classes[0]].keys())

        for idx, c in enumerate(lst_classes):
            df.loc[idx] = '0' # Initialize row
            df.loc[idx]['class'] = c

            if c != 'accuracy':
                for m in lst_metrics:
                    if m not in ['support']:
                        lst_values = list()
                        for d in lst_dcts:
                            lst_values.append(d[c][m])

                        df.loc[idx][m+'-mean'] = np.mean(lst_values)
                        df.loc[idx][m+'-std'] = np.std(lst_values)
                    else:
                        df.loc[idx][m] = d[c][m]
            else:
                lst_values = list()
                for d in lst_dcts:
                    lst_values.append(d[c])
                df.loc[idx]['precision-mean'] = np.mean(lst_values)
                df.loc[idx]['precision-std'] = np.std(lst_values)
                df.loc[idx]['runs'] = num_runs
                
        for c in column_names[1:]:
            df[c] = pd.to_numeric(df[c])

        return df

    # Gather all target result pkl files
    dct_target_results = defaultdict(lambda: defaultdict(defaultdict))

    for path in Path(target_folder).rglob('test_results_overall.pkl'):
        filename = path.name

        run = str(path.parent).split('/')[-1]
        dct_target_results[run][filename] = path

    # Generate all averaged files
    lst_runs = list(dct_target_results.keys())
    lst_filetypes = list(dct_target_results[lst_runs[0]].keys())

    # Process the target result files
    for filetype in lst_filetypes:
        lst_rel_files = []
        for run in lst_runs:
            lst_rel_files.append(dct_target_results[run][filetype])
        df = process_files(lst_rel_files)

        target_filename = os.path.join(target_folder, 'avg_' + lst_rel_files[0].stem)
        df.to_csv(target_filename + '.csv', float_format='%.4f')
        joblib.dump(df, target_filename + '.pkl')

def setup_default_logging(output_dir, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
        
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, output_dir