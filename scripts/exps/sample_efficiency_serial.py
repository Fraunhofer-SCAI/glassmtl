"""
Sample efficiency experiments
"""

from scripts.data_utils import seed_everything
seed_everything()


from typing import List, Tuple, Optional

import torch
import pytorch_lightning as pl
import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

from scripts.exps import data_splitting
from scripts import model, params, data_utils


SEED = params.SEED
pd.options.mode.chained_assignment = None  # default='warn'
torch.autograd.set_detect_anomaly(True)


def train_size_ablation(
        src_paths: List[str],
        result_path: str,
        prop_groups: List[List[str]],
        prop_ranges: List[Tuple[float]],
        prop_accuracies: List[int],
        n_folds: int,
        train_fracs: List[float],
        exps: List[str],
        scalers: List,
        with_t: bool = False,
        t_range: Optional[List[float]] = None,
        random_state: int = SEED,
        generate_sets: bool = True,
        max_epochs: int = 200
        ) -> None:
    """
    Split the source datasets into training, validation, and test sets.
    Then perform a sample efficiency study: Fit a model to each training set 
    and evaluate on the test set.

    :param src_paths: Paths to source datasets
    :param result_path: Path where the results are saved to.
    :param prop_groups: Groups of properties that are that are 
        treated together.
        Example: [[Viscosity500C, Viscosity600C], [Tg]]
    :param_ranges: (min,max)-Tuples of property values corresponding 
        to :obj:`prop_groups`
    :param prop_accuracies: List of decimal prop_accuracies of each 
        property except the target property.
    :param n_folds: Number of test splits.
    :param train_fracs: List of training fractions in %.
    :param exps: List of experimental setups: 
        single-task learning ("only_p1_single"), 
        non-random MTL ("aux_p_pad"), or 
        random MTL ("aux_p_random_pad")
    :param scalers: List of scalers to scale data of each property.
    :param with_t: Boolean indicating whether to use t-parametrization 
        or not.
    :param random_state: Random state for reproducability.
    :param generate_sets: Boolean indicating whether new training, 
        validation, and test sets should be generated.
    :param max_epochs: Maximum number of training epochs.
    """
    
    dg = data_splitting.DataSplit(
        src_paths,
        result_path,
        prop_ranges,
        prop_accuracies,
        n_folds,
        train_fracs,
        random_state
        )

    if generate_sets:
        dg.gen_train_val_test_sets(exps)
    
    for i in range(n_folds):
        for exp in exps:
            if 'only_p1' in exp:
                my_prop_groups = [prop_groups[0]]
            else:
                my_prop_groups = prop_groups

            n_props = len(list(itertools.chain.from_iterable(my_prop_groups)))

            for j, frac in enumerate(train_fracs):
                print('\nexp, with_t ', exp, with_t, 
                      ' test_split, frac: ', i, frac, '\n')
                pl.seed_everything(seed=random_state, workers=True)
                os.makedirs(
                    result_path+f'./test_{i}/{exp}/results', 
                    exist_ok=True
                    )
                mod = model.GlassAttentionNet(
                    n_props=n_props,
                    test_result_path=result_path\
                        +f'./test_{i}/{exp}/results/test_{frac}.csv',
                    predict_result_path=result_path\
                        +f'./test_{i}/{exp}/results/predict_{frac}.csv',
                    with_t = with_t,
                    loss = 'l1_single'
                    )
                
                checkpoint_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=result_path+f'./test_{i}/{exp}/models',
                    filename=f'{frac}',
                    monitor="val_loss_epoch",
                    save_top_k=1,
                    mode = 'min'
                    )

                early_stopping = pl.callbacks.EarlyStopping(
                    monitor="val_loss_epoch",
                    patience=40,
                    mode="min"
                    )    

                callbacks = [checkpoint_callback, early_stopping]

                trainer = pl.Trainer(
                    max_epochs = max_epochs,
                    enable_progress_bar=True,
                    enable_model_summary=True,
                    callbacks=callbacks, 
                    accelerator="gpu",
                    devices=1,
                    log_every_n_steps=1,
                    deterministic=True
                    )

                dm = data_utils.GlassDataModule(
                    scalers = scalers, 
                    prop_groups = my_prop_groups,
                    with_t = with_t,
                    t_range = t_range,
                    train_path=result_path\
                        +f'./test_{i}/{exp}/data/train_{frac}.csv',
                    val_path=result_path+f'./test_{i}/{exp}/data/val.csv',
                    test_path=result_path+f'./test_{i}/{exp}/data/test.csv',
                    # Predict on validation set
                    predict_path=result_path+f'./test_{i}/{exp}/data/val.csv'
                    )

                trainer.fit(mod, dm)
                trainer.test(mod, dm, ckpt_path = 'best')
                # Predict on val set to track performance 
                # of best perfomring model (optional)
                #trainer.predict(mod, dm, ckpt_path = 'best')

                # Save best model
                mod = model.GlassAttentionNet.load_from_checkpoint(
                    checkpoint_callback.best_model_path)
                torch.save(mod, result_path+f'./test_{i}/{exp}/model.pt')    


if __name__ == '__main__':
    from datetime import datetime
    import sys

    SEED = params.SEED
    pl.seed_everything(seed = SEED, workers = True)

    # Current date and time for timestamp
    # dd_mm_YY_HH:MM
    now = datetime.now().strftime("%d_%m_%Y_%H:%M")
    train_fracs = [1,2,4,8,16,32,64,100]
    n_folds = 5
    
    if sys.argv[1] == '--LogViscosity_Tg':
        # LogVisc+Tg
        train_size_ablation(
            src_paths = ['datasets/LogViscosity/all.csv', 
                         'datasets/Tg/all.csv'],
            result_path = f'results/{now}/LogViscosity_Tg/',
            prop_groups = [['LogViscosity773K', 'LogViscosity873K', 
                'LogViscosity973K', 'LogViscosity1073K', 
                'LogViscosity1173K', 'LogViscosity1273K',
                'LogViscosity1373K', 'LogViscosity1473K', 
                'LogViscosity1573K', 'LogViscosity1673K', 
                'LogViscosity1773K', 'LogViscosity1873K',
                'LogViscosity2073K', 'LogViscosity2273K', 
                'LogViscosity2473K'],['Tg']],
            prop_ranges = [(378, 1258)],
            prop_accuracies=[0],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = ['aux_p_pad', 'only_p1_single', 'aux_p_random_pad'],
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = False,
            random_state = SEED,
            generate_sets=True
            )
    
    elif sys.argv[1] == '--LogViscosity_Tg_t':
        # LogVisc+Tg with t
        train_size_ablation(
            src_paths = ['datasets/LogViscosity/all_t.csv', 
                         'datasets/Tg/all_t.csv'],
            result_path = f'results/{now}/LogViscosity_Tg_t/',
            prop_groups = [['LogViscosity'],['Tg']],
            prop_ranges = [(378, 1258)],
            prop_accuracies=[0],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = ['aux_p_pad', 'only_p1_single', 'aux_p_random_pad'],
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = True,
            t_range = [773, 2473],
            random_state = SEED,
            generate_sets=True
            )
        
    elif sys.argv[1] == '--YM_Density':
        # YoungsModulus293K+Density293K
        train_size_ablation(
            src_paths = ['datasets/YoungsModulus293K/all.csv', 
                        'datasets/Density293K/all.csv'],
            result_path = f'results/{now}/YoungsModulus293K_Density293K/',
            prop_groups = [['YoungsModulus293K'], ['Density293K']],
            prop_ranges = [(1.64, 8.5)],
            prop_accuracies=[2],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = ['aux_p_pad', 'only_p1_single', 'aux_p_random_pad'],
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = False,
            random_state = SEED,
            generate_sets=True,
            )
    
    elif sys.argv[1] == '--YM_Density_RI':
        # YoungsModulus293K+Density293K+RefractiveIndex
        train_size_ablation(
            src_paths = ['datasets/YoungsModulus293K/all.csv', 
                        'datasets/Density293K/all.csv',
                        'datasets/RefractiveIndex/all.csv'],
            result_path = f'results/{now}/\
YoungsModulus293K_Density293K_RefractiveIndex/',
            prop_groups = [['YoungsModulus293K'], 
                            ['Density293K'], 
                            ['RefractiveIndex']],
            prop_ranges = [(1.64, 8.5), (1.4, 2.75)],
            prop_accuracies=[2, 2],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = ['aux_p_pad', 'only_p1_single', 'aux_p_random_pad'],
            scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler()],
            with_t = False,
            random_state = SEED,
            generate_sets=True
            )
    
    else:
        raise ValueError('Wrong command line argument.')