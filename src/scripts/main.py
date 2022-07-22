from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine

import torch.nn as nn

import torchvision.utils as vutils

from tqdm import tqdm
from sklearn import metrics

import pandas as pd
import traceback
import time

import configparser
from optparse import OptionParser

import optuna
import joblib

from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_slice

import os
import random
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../modules/")

from dataset_selector import dataset_selector
from model_selector import model_selector
from optimizer_selector import optimizer_selector, lr_schedule_selector
from loss_selector import loss_selector
from utils.plotting_utils import *
from utils.visualization_utils import *
from transform_selector import select_transform, normalize
from utils.utils import make_dir_if_not_exists, save_evaluation_results, average_n_run_results, average_n_run_results_eval, setup_default_logging
from apr_p import select_apr_p

if os.environ.get('SLURM_NODEID') is not None:
    NUM_CPUS = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE'))
else:
    NUM_CPUS = os.cpu_count()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIR_RESULTS = '../../results'


def dct2df(dct, labels):

    new_dict = {}

    metrics = ['precision', 'recall', 'f1-score', 'support']

    lst_labels = []
    lst_metrics = []
    lst_values = []

    for label in labels:
        for metric in metrics:

            lst_labels.append(label)
            lst_metrics.append(metric)
            lst_values.append(dct[label][metric])

    new_dict['metric'] = lst_metrics
    new_dict['value'] = lst_values
    new_dict['class'] = lst_labels

    df = pd.DataFrame.from_dict(new_dict)
    return df


def dct2metrics(dct, labels):
    lst_precision = []
    lst_recall = []
    lst_f1score = []

    for label in labels:
            lst_precision.append(dct[label]['precision'])
            lst_recall.append(dct[label]['recall'])
            lst_f1score.append(dct[label]['f1-score'])

    return lst_precision, lst_recall, lst_f1score

def evaluate(config):

    setup_name = config['dataset_parameters']['preprocessing']

    frequency_preprocessing, kwargs = select_transform(config)

    test_transform = Compose([
        Resize((int(config['dataset_parameters']['image_size']), int(config['dataset_parameters']['image_size']))),
        frequency_preprocessing(**kwargs),
        normalize()
    ])
    
    train_dataset = dataset_selector(dataset_name=config['dataset_parameters']['train_dataset'], 
                                mode='train', 
                                transforms=test_transform, 
                                seed=42, 
                                basepath=config['dataset_parameters']['data_root'],
                                target_label=config['dataset_parameters']['derm7pt_target'],
                                augmentation_name=config['dataset_parameters']['target_augmentation']
                                )

    test_dataset = dataset_selector(dataset_name=config['dataset_parameters']['eval_dataset'], 
                                mode='test', 
                                transforms=test_transform, 
                                seed=42, 
                                basepath=config['dataset_parameters']['data_root'],
                                target_label=config['dataset_parameters']['derm7pt_target'],
                                augmentation_name=config['dataset_parameters']['target_augmentation']
                                )

    test_dl = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=int(config['evaluation_parameters']['batch_size']),
                                            shuffle=False,
                                            sampler=None,
                                            num_workers=int(config['hardware_parameters']['num_workers']),
                                            pin_memory=True)

    _, test_apr_p_transform = select_apr_p(setup_name)

    config.set('dataset_parameters', 'num_classes', str(train_dataset.get_num_classes()))

    lst_class_names = train_dataset.get_lst_class_names()

    # Create folder structure
    model_dir = os.path.join(DIR_RESULTS, config['experiment_parameters']['exp_dir'], config['experiment_parameters']['run_comment'])
    #dir_results = "/".join(config['evaluation_parameters']['model_path'].split('/')[:-1])

    eval_dataset_suffix = config['dataset_parameters']['target_augmentation']
    eval_dataset_suffix = '_' + eval_dataset_suffix if eval_dataset_suffix != '' else ''
    dir_results = os.path.join(model_dir, config['dataset_parameters']['eval_dataset'].replace('_', '-') + eval_dataset_suffix)
    make_dir_if_not_exists(dir_results)

    logger.info("")
    logger.info('--- Building Network ---')
    model = model_selector(model_name=config['model_parameters']['model_type'], num_classes=int(config['dataset_parameters']['num_classes']))

    model_path = os.path.join(model_dir, 'best_checkpoint.pth')
    logger.info('Loading model from %s'%model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Test the model
    with torch.no_grad():
        model.to(config['hardware_parameters']['device'])
        model.eval()

        correct = 0
        total = 0

        all_labels = []
        all_predicted = []
        all_sm_predictions = []

        for step, (images, labels) in enumerate(test_dl):
            images = images.to(config['hardware_parameters']['device'])
            labels = labels.to(config['hardware_parameters']['device'])

            # Mixes/Transforms batch if apr-p variant is selected
            images = test_apr_p_transform(images)
            
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().data.numpy()

            all_labels.append(labels)
            all_predicted.append(predicted)
            all_sm_predictions.append(nn.functional.softmax(outputs, dim=0).data)

        test_accuracy = 100 * np.true_divide(correct, total)
        logger.info('Accuracy of the model on the test images: %.2f%%' % test_accuracy)

    all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
    all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy().astype(int)
    all_sm_predictions = torch.cat(all_sm_predictions, dim=0).cpu().numpy()

    # Handle the case that predictions do not contain all labels from GT.
    lst_predicted_labels = np.unique(all_predicted)
    lst_predicted_labels = np.array(lst_class_names)[lst_predicted_labels]
    #lst_class_names = None if len(lst_predicted_labels) != len(lst_class_names) else lst_class_names

    print(config['dataset_parameters']['eval_dataset'])

    save_evaluation_results(all_predicted, all_labels, dir_results, target_names=lst_class_names)

    try:

        _ = plot_confusion_matrix(all_labels, all_predicted, lst_class_names, dest_path=os.path.join(dir_results, 'plot_confusion-matrix.png'))

        dct_classification_report = metrics.classification_report(all_labels, all_predicted, target_names=lst_class_names, output_dict=True)
        
        with open(os.path.join(dir_results, 'results.txt'), 'w') as f:
            f.write(metrics.classification_report(all_labels, all_predicted, target_names=lst_class_names, output_dict=False))
        
        df_classification_report = dct2df(dct_classification_report, lst_class_names)
        df_classification_report.to_csv(os.path.join(dir_results, 'results.txt'), sep=',')
        lst_precision, lst_recall, lst_f1score = dct2metrics(dct_classification_report, lst_class_names)

        # Plot recall and precision per class
        _ = plot_pr(lst_precision, lst_recall, lst_class_names, dest_path=os.path.join(dir_results, 'plot_pr-curve.png'))

        # Plot F1 score
        _ = plot_pr(lst_f1score, lst_f1score, lst_class_names, lst_metric_names=('F1-Score', 'F1-Score'), dest_path=os.path.join(dir_results, 'plot_f1-score.png'))
    except Exception:
        traceback.print_exc()
        print('Evaluation error')

    
def train_evaluate(trial, config, train_args):

    logger.info(train_args)

    num_epochs = int(config['training_parameters']['num_epochs'])
    setup_name = config['dataset_parameters']['preprocessing']

    start = time.time()

    frequency_preprocessing, kwargs = select_transform(config)

    train_transform = Compose([
        RandomResizedCrop((int(config['dataset_parameters']['image_size']), int(config['dataset_parameters']['image_size'])), scale=(0.4, 1.0)),
        RandomAffine(degrees=90, scale=(0.8,1.2), shear=(-20,20)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        frequency_preprocessing(**kwargs),
        normalize()
    ])
    test_transform = Compose([
        Resize((int(config['dataset_parameters']['image_size']), int(config['dataset_parameters']['image_size']))),
        frequency_preprocessing(**kwargs),
        normalize()
    ])


    train_dataset = dataset_selector(dataset_name=config['dataset_parameters']['train_dataset'], 
                                mode='train', 
                                transforms=train_transform, 
                                seed=42, 
                                basepath=config['dataset_parameters']['data_root'],
                                target_label=config['dataset_parameters']['derm7pt_target'],
                                augmentation_name=config['dataset_parameters']['target_augmentation']
                                )

    val_dataset = dataset_selector(dataset_name=config['dataset_parameters']['train_dataset'], 
                                mode='val', 
                                transforms=test_transform, 
                                seed=42, 
                                basepath=config['dataset_parameters']['data_root'],
                                target_label=config['dataset_parameters']['derm7pt_target'],
                                augmentation_name=config['dataset_parameters']['target_augmentation']
                                )
    
    test_dataset = dataset_selector(dataset_name=config['dataset_parameters']['train_dataset'], 
                                mode='test', 
                                transforms=test_transform, 
                                seed=42, 
                                basepath=config['dataset_parameters']['data_root'],
                                target_label=config['dataset_parameters']['derm7pt_target'],
                                augmentation_name=config['dataset_parameters']['target_augmentation']
                                )

    if config.getboolean('dataset_parameters', 'weighted_sampling'):
        gen = torch.Generator()
        gen = gen.manual_seed(2147483647)

        weightedSampler = torch.utils.data.WeightedRandomSampler(train_dataset.get_sample_weights(),
                                                                len(train_dataset), 
                                                                replacement=True, 
                                                                generator=gen)
        shuffleTrain = False
    else:
        weightedSampler = None
        shuffleTrain = True

    train_dl = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=int(config['training_parameters']['batch_size']),
                                            shuffle=shuffleTrain,
                                            sampler=weightedSampler,
                                            num_workers=int(config['hardware_parameters']['num_workers']),
                                            pin_memory=True)

    val_dl = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=int(config['training_parameters']['batch_size']),
                                            shuffle=False,
                                            sampler=None,
                                            num_workers=int(config['hardware_parameters']['num_workers']),
                                            pin_memory=True)
    
    test_dl = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=int(config['training_parameters']['batch_size']),
                                            shuffle=False,
                                            sampler=None,
                                            num_workers=int(config['hardware_parameters']['num_workers']),
                                            pin_memory=True)

    train_apr_p_transform, test_apr_p_transform = select_apr_p(setup_name)

    config.set('dataset_parameters', 'num_classes', str(train_dataset.get_num_classes()))

    train_args = {
        'lr': float(config['training_parameters']['lr']),
        'wd':float(config['training_parameters']['wd']),
        'momentum': float(config['training_parameters']['momentum'])
    }

    lst_class_names = train_dataset.get_lst_class_names()

    # Create folder structure
    dir_results = os.path.join(DIR_RESULTS, config['experiment_parameters']['exp_dir'], config['experiment_parameters']['run_comment'])
    best_checkpoint_path = os.path.join(dir_results, 'best_checkpoint.pth')
    last_checkpoint_path = os.path.join(dir_results, 'last_checkpoint.pth')
    make_dir_if_not_exists(dir_results)

    writer = SummaryWriter(dir_results)

    logger.info("")
    logger.info('--- Building Network ---')
    model = model_selector(model_name=config['model_parameters']['model_type'], num_classes=int(config['dataset_parameters']['num_classes']))
    model.to(config['hardware_parameters']['device'])

    # Loss and optimizer
    criterion = loss_selector(config['training_parameters']['loss'])
    optimizer = optimizer_selector(name=config['training_parameters']['optimizer'],
                                   learning_rate=train_args['lr'],
                                   wd=train_args['wd'],
                                   momentum=train_args['momentum'],
                                   model=model)
    scheduler = lr_schedule_selector(
        optimizer=optimizer, 
        name=config['scheduler_parameters']['lr_scheduler'],
        plateau_patience=int(config['scheduler_parameters']['lr_plateau_patience']),
        plateau_factor=float(config['scheduler_parameters']['lr_plateau_factor']),
        step_size=int(config['scheduler_parameters']['lr_step_size']),
        step_gamma=float(config['scheduler_parameters']['lr_step_gamma']),
        )

    global_step = 0
    ES_counter = 0
    total = 0
    correct = 0
    min_loss = sys.float_info.max

    # Train the model
    logger.info("")
    logger.info('--- Training Network ---')
    for epoch in range(num_epochs):
        model.train()
        # Convert numpy arrays to torch tensors
        for step, (images, labels) in enumerate(tqdm(train_dl)):
                
            images = images.to(config['hardware_parameters']['device'], non_blocking=True)
            labels = labels.to(config['hardware_parameters']['device'], non_blocking=True)            

            # Mixes/Transforms batch if apr-p variant is selected
            images = train_apr_p_transform(images)

            # Forward pass
            outputs = model(images.float())

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (labels == predicted.squeeze()).float().mean()

            if (global_step+1) % int(config['logging_parameters']['logging_freq']) == 0:
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                writer.add_scalar('training/loss', loss.item(), global_step)
                writer.add_scalar('training/accuracy', accuracy.item(), global_step)

                if config.getboolean('logging_parameters', 'log_images'):
                    x = vutils.make_grid(images, normalize=False, scale_each=True)
                    writer.add_image('Image', x, global_step, dataformats='CHW')

            global_step += 1

        # Validate
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        all_sm_predictions = []

        running_val_loss = 0.0
        running_correct = 0.0

        with torch.no_grad():
            model.eval()
            for step, (images, labels) in enumerate(val_dl):
                images = images.to(config['hardware_parameters']['device'], non_blocking=True)
                labels = labels.to(config['hardware_parameters']['device'], non_blocking=True)

                # Mixes/Transforms batch if apr-p variant is selected
                images = test_apr_p_transform(images)

                # Forward pass
                outputs = model(images.float())

                total += labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()
                running_val_loss += criterion(outputs, labels)

                all_labels.append(labels)
                all_predicted.append(predicted)
                all_sm_predictions.append(nn.functional.softmax(outputs, dim=0).data)

            all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
            all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy().astype(int)
            all_sm_predictions = torch.cat(all_sm_predictions, dim=0).cpu().numpy()

            val_loss = running_val_loss.item() / total
            val_accuracy = 100. * running_correct / total

            if scheduler is not None:
                scheduler.step(val_loss)

            logger.info('Validation:')
            logger.info('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Acc: {:.4f}, ESC: {}/{}\n'.format(epoch + 1, int(config['training_parameters']['num_epochs']), global_step + 1,
                                                                               val_loss, val_accuracy,
                                                                               ES_counter, int(config['training_parameters']['early_stopping_patience'])))
            # info = {'loss': val_loss.item(), 'accuracy': accuracy.item(), 'LR': optimizer.param_groups[0]['lr']}
            writer.add_scalar('validation/loss', val_loss, epoch)
            writer.add_scalar('validation/accuracy', val_accuracy, epoch)
            writer.add_scalar('hyperparam/LR', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('hyperparam/ES', ES_counter, epoch)
            for i in range(int(config['dataset_parameters']['num_classes'])):
                i_index = (all_labels == i)
                labels = np.zeros_like(all_labels)
                labels[i_index] = 1
                predictions = all_sm_predictions[:, i]
                writer.add_pr_curve('%s/pr_curve' % lst_class_names[i], labels, predictions, global_step=global_step)

            # Check if early stopping applies
            if int(config['training_parameters']['early_stopping_patience']) != -1:
                if val_loss < min_loss:
                    min_loss = val_loss
                    ES_counter = 0

                    save_obj = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'args': config,
                    }
                    torch.save(save_obj, best_checkpoint_path)

                else:

                    if ES_counter >= int(config['training_parameters']['early_stopping_patience']):
                        break

                    ES_counter += 1

            if int(config['optuna_parameters']['optuna_runs']) != 0:
                # Report intermediate objective value.
                intermediate_value = val_accuracy
                trial.report(intermediate_value, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

    # Save final model
    save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'args': config,
        }
    #torch.save(save_obj, last_checkpoint_path)

    logger.info('Loading best model from %s'%best_checkpoint_path)
    checkpoint = torch.load(best_checkpoint_path)
    msg = model.load_state_dict(checkpoint['model'], strict=False)

    global_step = 0

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        model.to(config['hardware_parameters']['device'])
        model.eval()

        correct = 0
        total = 0

        all_labels = []
        all_predicted = []
        all_sm_predictions = []

        for step, (images, labels) in enumerate(test_dl):
            images = images.to(config['hardware_parameters']['device'])
            labels = labels.to(config['hardware_parameters']['device'])

            # Mixes/Transforms batch if apr-p variant is selected
            images = test_apr_p_transform(images)

            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().data.numpy()

            all_labels.append(labels)
            all_predicted.append(predicted)
            all_sm_predictions.append(nn.functional.softmax(outputs, dim=0).data)

        test_accuracy = 100 * np.true_divide(correct, total)
        logger.info('Accuracy of the model on the test images: %.2f%%' % test_accuracy)

    all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
    all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy().astype(int)
    all_sm_predictions = torch.cat(all_sm_predictions, dim=0).cpu().numpy()

    save_evaluation_results(all_predicted, all_labels, dir_results, target_names=lst_class_names)

    # Write evaluation and hyperparameters to tensorboard
    for i in range(int(config['dataset_parameters']['num_classes'])):
        i_index = (all_labels == i)
        labels = np.zeros_like(all_labels)
        labels[i_index] = 1
        predictions = all_sm_predictions[:, i]
        writer.add_pr_curve('%s/pr_curve' % lst_class_names[i], labels, predictions, global_step=global_step)

        fig_roc = plot_roc_curve(labels, predictions)
        writer.add_figure('ROC Curves/%s' % lst_class_names[i], fig_roc, global_step=global_step, close=True, walltime=None)


    fig_cm = plot_confusion_matrix(all_labels, all_predicted, lst_class_names)
    writer.add_figure('Test Evaluation/Confusion Matrix', fig_cm, global_step=global_step, close=True, walltime=None)

    dct_classification_report = metrics.classification_report(all_labels, all_predicted, target_names=lst_class_names, output_dict=True)
    
    with open(os.path.join(dir_results, 'results.txt'), 'w') as f:
        f.write(metrics.classification_report(all_labels, all_predicted, target_names=lst_class_names, output_dict=False))
    
    df_classification_report = dct2df(dct_classification_report, lst_class_names)
    df_classification_report.to_csv(os.path.join(dir_results, 'results.txt'), sep=',')
    lst_precision, lst_recall, lst_f1score = dct2metrics(dct_classification_report, lst_class_names)

    # Plot recall and precision per class
    fig_pr = plot_pr(lst_precision, lst_recall, lst_class_names)
    writer.add_figure('Test Evaluation/Precision & Recall', fig_pr, global_step=global_step, close=True, walltime=None)

    # Plot F1 score
    fig_f1 = plot_pr(lst_f1score, lst_f1score, lst_class_names, lst_metric_names=('F1-Score', 'F1-Score'))
    writer.add_figure('Test Evaluation/F1Score & Accuracy', fig_f1, global_step=global_step, close=True, walltime=None)

    end = time.time()
    logger.info("Total time: %.4f"%(end-start))

    #os.remove(best_checkpoint_path)

    return val_accuracy

def create_hpspace(trial):
    # Use trial to create your hyper parameter space based 
    # based on any conditon or loops !!

    train_args = {
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-1),
        'wd': float(config['training_parameters']['wd']),
        'momentum': trial.suggest_uniform('momentum', 0.0, 1.0)
    }
    objective = train_evaluate(trial, config, train_args)

    return objective

        
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-f", "--config", dest="config", default='config.ini', help="config file to be used", metavar="FILE")
    parser.add_option("-n", '--random-runs', type=int, default=1)
    parser.add_option('--random-runs-start-from', type=int, default=0)
    parser.add_option("-o", '--optuna-runs', type=int, default=0)
    parser.add_option("-d", "--data-root", default='/home/User/Datasets')
    (options, args) = parser.parse_args()

    # Assign config file
    CONFIG_PATH = options.config
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    config.set('dataset_parameters','data_root', options.data_root)
    config.add_section('hardware_parameters')
    config.set('hardware_parameters','num_workers', str(NUM_CPUS))
    config.set('hardware_parameters','device', str(DEVICE))
    config.add_section('optuna_parameters')
    config.set('optuna_parameters','optuna_runs', str(options.optuna_runs))

    logger, output_dir = setup_default_logging(os.path.join(DIR_RESULTS, config['experiment_parameters']['exp_dir']))
    logger.info(options)
    logger.info("")
    logger.info("#######################")
    logger.info("# Hardware Parameters #")
    logger.info("#######################")
    logger.info('Num_CPUs: %s'%NUM_CPUS)
    logger.info('Device: %s\n'%DEVICE)

    if config['experiment_parameters']['mode'] == 'test':
        base_run_comment = config['experiment_parameters']['run_comment']

        # If a list of multiple eval datasets is given, split list and iterate.
        if ',' in config['dataset_parameters']['eval_dataset']:
            eval_datasets = config['dataset_parameters']['eval_dataset'].split(',')
        else:
            eval_datasets = [config['dataset_parameters']['eval_dataset']]
        
        for eval_dataset in eval_datasets:
            config['dataset_parameters']['eval_dataset'] = eval_dataset
            
            for run_idx in range(options.random_runs): 
                config['experiment_parameters']['run_comment'] = base_run_comment + f'_run-{run_idx}'
                evaluate(config)

            eval_dataset_suffix = config['dataset_parameters']['target_augmentation']
            eval_dataset_suffix = '_' + eval_dataset_suffix if eval_dataset_suffix != '' else ''

            average_n_run_results_eval(
                target_folder=os.path.join(DIR_RESULTS, config['experiment_parameters']['exp_dir']), 
                target_dataset=config['dataset_parameters']['eval_dataset'].replace('_', '-') + eval_dataset_suffix, 
                num_runs=options.random_runs, 
                run_comment=base_run_comment
            )

    elif config['experiment_parameters']['mode'] == 'train':
        
        if options.optuna_runs != 0:
            # Hyperparameter Optimization

            config['experiment_parameters']['run_comment'] = config['experiment_parameters']['run_comment'] + f'_optuna'

            study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize', pruner=optuna.pruners.HyperbandPruner())
            study.optimize(create_hpspace, n_trials=options.optuna_runs)

            pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

            logger.info("\n\n")
            logger.info("Study statistics:")
            logger.info("  Number of finished trials: %d"%len(study.trials))
            logger.info("  Number of pruned trials: %d"%len(pruned_trials))
            logger.info("  Number of complete trials: %d"%len(complete_trials))

            trial = study.best_trial
            best = str(trial.value)
            logger.info("Best trial:")
            logger.info("  Value: %s\n"%best)
            
            df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
            logger.info(df.head())

            joblib.dump(study, os.path.join(output_dir, 'optuna.pkl'))

            fig = plot_contour(study)
            fig.write_image(os.path.join(output_dir, 'contour.png'))
            
            fig = plot_parallel_coordinate(study)
            fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))

            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(output_dir, 'history.png'))
            
            fig = plot_slice(study)
            fig.write_image(os.path.join(output_dir, 'slice.png'))
        else:

            base_run_comment = config['experiment_parameters']['run_comment']

            for run_idx in range(options.random_runs_start_from, options.random_runs):
                
                logger.info("")
                logger.info(20*"=")
                logger.info(f"Starting Run {run_idx}")
                logger.info(20*"=")

                # Reproducibility
                torch.manual_seed(run_idx)
                random.seed(run_idx)
                np.random.seed(run_idx)
                #torch.use_deterministic_algorithms(True)

                config['experiment_parameters']['run_comment'] = base_run_comment + f'_run-{run_idx}'

                train_args = {
                    'lr': float(config['training_parameters']['lr']),
                    'wd': float(config['training_parameters']['wd']),
                    'momentum': float(config['training_parameters']['momentum'])
                }

                # Normal Training
                train_evaluate(None, config, train_args)
            
            average_n_run_results(output_dir)

