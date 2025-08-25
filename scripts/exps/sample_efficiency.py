"""
Sample efficiency experiments
"""

from scripts.data_utils import seed_everything
seed_everything()

from typing import List, Tuple
import itertools
import subprocess
import time
from scripts.exps import data_splitting
from scripts import params


def data_split(
        src_paths: List[str],
        result_path: str,
        prop_ranges: List[Tuple[float]],
        prop_accuracies: List[int],
        n_folds: int,
        train_fracs: List[float],
        exps: List[str],
        random_state: int = params.SEED,
        generate_sets: bool = True
        ) -> None:
    """
    Split the source datasets into training, validation, and test sets.
    Then perform a sample efficiency study: Fit a model to each training set 
    and evaluate on the test set.

    :param src_paths: Paths to source datasets
    :param result_path: Path where the results are saved to.
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
    :param random_state: Random state for reproducability.
    :param generate_sets: Boolean indicating whether new training, 
        validation, and test sets should be generated.
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



def mult_proc(
        ident: str, 
        prop: str,
        train_fracs: List[int],
        n_folds: int,  
        exps: List[str], 
        max_n_processes = params.MAX_N_PROCESSES,
        ) -> None:
    """Run multiple processes in parallel."""

    processes = []
    for frac, fold, exp in list(
        itertools.product(train_fracs, list(range(n_folds)), exps)
        ):
        p = subprocess.Popen(
            ['python', 
            '-m',
            'scripts.exps.train_model', 
            f'--prop={prop}',
            f'--ident={ident}',
            f'--train_frac={frac}',
            f'--fold={fold}',
            f'--exp={exp}']
            )
        processes.append(p)

        if len(processes) >= max_n_processes:
            while True:
                for p in processes:
                    if p.poll() is not None: 
                        processes.remove(p)
                        break
                else:
                    time.sleep(1) 
                    continue
                break

    for p in processes:
        p.wait()



if __name__ == '__main__':
    from datetime import datetime
    import sys

    train_fracs = [100, 64, 32, 16, 8, 4, 2, 1]
    # Three settings: single-task learning, 
    # non-random multitask learning, random multitask learning
    exps = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']
    n_folds = 5
    # Current date and time for timestamp dd_mm_YY_HH:MM:SS
    ident = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") 
          
    prop = sys.argv[1]

    
    if sys.argv[1] == 'LogViscosity_Tg':
        # LogVisc+Tg
        data_split(
            src_paths = ['datasets/LogViscosity/all.csv', 
                         'datasets/Tg/all.csv'],
            result_path = f'results/{ident}/\
LogViscosity_Tg/seed{params.SEED}/',
            prop_ranges = [(378, 1260)],
            prop_accuracies=[0],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = exps,
            generate_sets=True
            )
    
    elif sys.argv[1] == 'LogViscosity_Tg_t':
        # LogVisc+Tg with t
        data_split(
            src_paths = ['datasets/LogViscosity/all_t.csv', 
                         'datasets/Tg/all_t.csv'],
            result_path = f'results/{ident}/\
LogViscosity_Tg_t/seed{params.SEED}/',
            prop_ranges = [(378, 1260)],
            prop_accuracies=[0],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = exps,
            generate_sets=True
            )
        
    elif sys.argv[1] == 'YM_Density':
        # YoungsModulus293K+Density293K
        data_split(
            src_paths = ['datasets/YoungsModulus293K/all.csv', 
                        'datasets/Density293K/all.csv'],
            result_path = f'results/{ident}/\
YoungsModulus293K_Density293K/seed{params.SEED}/',
            prop_ranges = [(1.64, 8.49)],
            prop_accuracies=[2],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = exps,
            generate_sets=True
            )
    
    elif sys.argv[1] == 'YM_Density_RI':
        # YoungsModulus293K+Density293K+RefractiveIndex
        data_split(
            src_paths = ['datasets/YoungsModulus293K/all.csv', 
                        'datasets/Density293K/all.csv',
                        'datasets/RefractiveIndex/all.csv'],
            result_path = f'results/{ident}/\
YoungsModulus293K_Density293K_RefractiveIndex/seed{params.SEED}/',
            prop_ranges = [(1.64, 8.49), (1.4, 2.75)],
            prop_accuracies=[2, 2],
            n_folds = n_folds,
            train_fracs = train_fracs,
            exps = exps,
            generate_sets=True
            )
    
    else:
        raise ValueError('Wrong command line argument.')
    
    mult_proc(ident, prop, train_fracs, n_folds, exps)