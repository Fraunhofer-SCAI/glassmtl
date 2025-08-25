"""
Train and evaluate models
"""

from scripts.data_utils import seed_everything
seed_everything()

from typing import List, Optional

import torch
import pytorch_lightning as pl
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from scripts import model, params, data_utils

torch.autograd.set_detect_anomaly(True)
pd.options.mode.chained_assignment = None  # default='warn'


def train_size_ablation(
        result_path: str,
        prop_groups: List[List[str]],
        fold: int,
        train_frac: float,
        exp: str,
        scalers: List,
        with_t: bool = False,
        t_range: Optional[List[float]] = None,
        max_epochs: int = 200
        ) -> None:
    """
    Split the source datasets into training, validation, and test sets.
    Then perform a sample efficiency study: Fit a model to each training set 
    and evaluate on the test set.

    :param result_path: Path where the results are saved to.
    :param prop_groups: Groups of properties that are that are 
        treated together.
        Example: [[Viscosity500C, Viscosity600C], [Tg]]
    :param fold: Number of test splits.
    :param train_frac: List of training fractions in %.
    :param exp: List of experimental setups: 
        single-task learning ("only_p1_single"), 
        non-random MTL ("aux_p_pad"), or 
        random MTL ("aux_p_random_pad")
    :param scalers: List of scalers to scale data of each property.
    :param with_t: Boolean indicating whether to use t-parametrization 
        or not.
    :param max_epochs: Maximum number of training epochs.
    """
    
    
    if 'only_p1' in exp:
        my_prop_groups = [prop_groups[0]]
    else:
        my_prop_groups = prop_groups

    n_props = len(list(itertools.chain.from_iterable(my_prop_groups)))

    print('\nexp, with_t: ', exp, with_t, 
            ' test_split, frac: ', fold, train_frac, '\n')
    os.makedirs(
        result_path+f'./test_{fold}/{exp}/results', 
        exist_ok=True
        )
    if 'single' in exp:
        mod = model.GlassAttentionNet(
            n_props=n_props,
            test_result_path=result_path\
                +f'./test_{fold}/{exp}/results/test_{train_frac}.csv',
            predict_result_path=result_path\
                +f'./test_{fold}/{exp}/results/predict_{train_frac}.csv',
            with_t = with_t,
            loss = 'l1_single'    # Single-task loss
            )
    else:
        mod = model.GlassAttentionNet(
            n_props=n_props,
            test_result_path=result_path\
                +f'./test_{fold}/{exp}/results/test_{train_frac}.csv',
            predict_result_path=result_path\
                +f'./test_{fold}/{exp}/results/predict_{train_frac}.csv',
            with_t = with_t,
            loss = 'l1_multi'    # Multitask loss
            )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=result_path+f'./test_{fold}/{exp}/checkpoints',
        filename=f'{train_frac}',
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
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks, 
        accelerator="gpu",
        devices=1,
        logger = False,
        log_every_n_steps=1,
        deterministic=True
        )

    dm = data_utils.GlassDataModule(
        scalers = scalers, 
        prop_groups = my_prop_groups,
        with_t = with_t,
        t_range = t_range,
        train_path=result_path\
            +f'test_{fold}/{exp}/data/train_{train_frac}.csv',
        val_path=result_path+f'test_{fold}/{exp}/data/val.csv',
        test_path=result_path+f'test_{fold}/{exp}/data/test.csv',
        # Predict on validation set
        predict_path=result_path+f'test_{fold}/{exp}/data/val.csv'
        )

    trainer.fit(mod, dm)
    trainer.test(mod, dm, ckpt_path = 'best')
    ## Predict on val set to track performance 
    ## of best performing model (optional)
    #trainer.predict(mod, dm, ckpt_path = 'best')

    # Save best model
    mod = model.GlassAttentionNet.load_from_checkpoint(
        checkpoint_callback.best_model_path)
    torch.save(mod, result_path+f'test_{fold}/{exp}/best_model.pt')    

    ## Save parameters of multi-task loss function of 
    ## best trained model (optional)
    #loss_weights = mod.loss_weights.detach().cpu().numpy()
    #loss_weights = loss_weights.reshape(-1,1)
    #loss_terms = mod.loss_terms.detach().cpu().numpy()
    #loss_terms = loss_terms.reshape(-1,1)
    #params_path = result_path+f'test_{fold}/{exp}/results/params_{train_frac}.csv'
    #col_idx = pd.MultiIndex.from_product([[f'{train_frac}'], 
    #                                        ['weight', 'term']])
    #index = dm.ds_test.le_props.classes_
    #df_params = pd.DataFrame(np.hstack([loss_weights, 
    #                                    loss_terms]), 
    #                            index = index, 
    #                            columns = col_idx) 
    #df_params.to_csv(params_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prop", type=str, required=True)
    parser.add_argument("--ident", type=str, required=True)
    parser.add_argument("--train_frac", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()
    
    if args.prop == 'LogViscosity_Tg':
        # LogVisc+Tg
        train_size_ablation(
            result_path = f'results/{args.ident}/\
LogViscosity_Tg/seed{params.SEED}/',
            prop_groups = [['LogViscosity773K', 'LogViscosity873K', 
                'LogViscosity973K', 'LogViscosity1073K', 
                'LogViscosity1173K', 'LogViscosity1273K',
                'LogViscosity1373K', 'LogViscosity1473K', 
                'LogViscosity1573K', 'LogViscosity1673K', 
                'LogViscosity1773K', 'LogViscosity1873K',
                'LogViscosity2073K', 'LogViscosity2273K', 
                'LogViscosity2473K'],['Tg']],
            fold = args.fold,
            train_frac = args.train_frac,
            exp = args.exp,
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = False,
            )
    
    elif args.prop == 'LogViscosity_Tg_t':
        # LogVisc+Tg with t
        train_size_ablation(
            result_path = f'results/{args.ident}/\
LogViscosity_Tg_t/seed{params.SEED}/',
            prop_groups = [['LogViscosity'],['Tg']],
            fold = args.fold,
            train_frac = args.train_frac,
            exp = args.exp,
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = True,
            t_range = [773, 2473]
            )
        
    elif args.prop == 'YM_Density':
        # YoungsModulus293K+Density293K
        train_size_ablation(
            result_path = f'results/{args.ident}/\
YoungsModulus293K_Density293K/seed{params.SEED}/',
            prop_groups = [['YoungsModulus293K'], ['Density293K']],
            fold = args.fold,
            train_frac = args.train_frac,
            exp = args.exp,
            scalers = [MinMaxScaler(), MinMaxScaler()],
            with_t = False
            )
    
    elif args.prop == 'YM_Density_RI':
        # YoungsModulus293K+Density293K+RefractiveIndex
        train_size_ablation(
            result_path = f'results/{args.ident}/\
YoungsModulus293K_Density293K_RefractiveIndex/seed{params.SEED}/',
            prop_groups = [['YoungsModulus293K'], 
                            ['Density293K'], 
                            ['RefractiveIndex']],
            fold = args.fold,
            train_frac = args.train_frac,
            exp = args.exp,
            scalers = [MinMaxScaler(), MinMaxScaler(), MinMaxScaler()],
            with_t = False
            )