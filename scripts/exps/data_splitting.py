"""
Methods for data splitting
"""

from typing import List, Tuple

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from scripts import params

pd.options.mode.chained_assignment = None  # default='warn'


class DataSplit():
    """
    Class to merge data of multiple properties and perform subsequent 
    splitting into training set, validation set, and test set.
    """
    def __init__(
            self,
            src_paths: List[str],
            result_path: str,
            prop_ranges: List[Tuple[float]],
            prop_accuracies: List[int],
            n_folds: int = 5,
            train_fracs: List[int]=[100, 64, 32, 16, 8, 4, 2, 1],
            random_state: int = params.SEED
            ):
        """
        :param src_paths: List with paths to source files of each property.
            The order determines the order of the properties.
            The first property is the target property.
            The other properties are the 
        :param result_path: Path to folder with resulting files.
        :param prop_ranges: List of tuples with minimum and maximum values 
            for each property except the target property.    
        :param prop_accuracies: List of decimal prop_accuracies of each 
            property except the target property.
        :param scalers: List of scalers to scale data of each property.
        :param n_folds: Number of test splits.
        :param train_fracs: List with fractions in % indicating how many 
            samples are used to select the training sets.
        :param with_t: Boolean indicating whether to use t-parametrization 
            or not.
        :param random_state: Random state for reproducability.
        """
        super().__init__()

        self.df_src_list = [pd.read_csv(path, index_col = 0, header = [0,1]) 
            for path in src_paths]
        self.n_props = len(src_paths)
        self.result_path = result_path
        self.prop_ranges = prop_ranges
        self.prop_accuracies = prop_accuracies
        self.n_folds = n_folds
        self.train_fracs = train_fracs
        self.random_state = random_state

        # Determine property with maximum number of input elements
        idx_max = np.argmax([len(df.elements.columns)
            for df in self.df_src_list])
        self.df_max = self.df_src_list[idx_max]
        

    def pad(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pad `elements` and `at_fracs` cells of :obj:`df` to 
        maximum length.
        """
        df = pd.DataFrame(df, columns = self.df_max.columns)
        df['elements'] = df['elements'].astype(object)
        df.loc[:,['elements']] = df.loc[:,['elements']].fillna('X')
        df.loc[:,['at_fracs']] = df.loc[:,['at_fracs']].fillna(0.0)
        return df


    def only_p1(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    

    def aux_p(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all non-target property data to :obj:`df`.
        """
        df_all = self.pad(
            pd.concat([df, *self.df_src_list[1:]], axis = 0)
            )
        # Shuffle
        df_all = df_all.sample(frac=1, random_state = self.random_state)
        return df_all
    

    def aux_p_random(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all auxiliary non-target property data with random property 
        values to :obj:`df`.
        """
        rng = np.random.default_rng(self.random_state)
        df_list = []
        for i in range(self.n_props-1):
            range_min = self.prop_ranges[i][0]
            range_max = self.prop_ranges[i][1]
            val_random = np.round(rng.random(
                size = self.df_src_list[i+1].shape[0])\
                            *(range_max - range_min) + range_min, 
                self.prop_accuracies[i])
            df_random = self.df_src_list[i+1].copy()
            df_random.loc[:,('prop_val','0')] = val_random
            df_list.append(df_random)
        df_all = self.pad(pd.concat([df, *df_list], axis = 0))
        # Shuffle
        df_all = df_all.sample(frac=1, random_state=self.random_state)
        return df_all

    
    def test_splits(self, exp):
        """
        Generate splits into unscaled test data and unscaled test-complement 
        data stratified with respect to the property names.
        """
        # Shuffle
        df_src = self.df_src_list[0].sample(
            frac=1, 
            random_state=self.random_state
            )
        skf = StratifiedKFold(n_splits=self.n_folds)
        y_val = df_src[('prop_id','0')].values
        for i, (test_c_idx, test_idx) in enumerate(
            skf.split(X=df_src.values, 
            y=y_val)
            ):
            df_test = df_src.iloc[test_idx, :]
            df_test_c = df_src.iloc[test_c_idx, :]
            if 'pad' in exp:
                df_test = self.pad(df_test)
                df_test_c = self.pad(df_test_c)
            # Save files
            os.makedirs(
                self.result_path+f'./test_{i}/{exp}/data', 
                exist_ok=True
                )
            df_test.to_csv(
                self.result_path+f'./test_{i}/{exp}/data/test.csv'
                )
            df_test_c.to_csv(
                self.result_path+f'./test_{i}/{exp}/data/test_c.csv'
                )
        print('test splits done')


    def val_splits(self, nb_test, exp):
        """
        Generate (20%/80%) splits into unscaled validation and 
        validation-complement data stratified with respect to the property 
        names.
        """
        path = self.result_path+f'./test_{nb_test}/{exp}/data/test_c.csv'
        df_test_c = pd.read_csv(path, index_col = 0, header = [0,1])    
        df_test_c_prop = df_test_c[('prop_id','0')]
        val_c_idx, val_idx = train_test_split(
            np.arange(len(df_test_c.index)),
            test_size = 0.2, 
            random_state = self.random_state,
            shuffle = True,
            stratify = df_test_c_prop
            )
        df_val = df_test_c.iloc[val_idx,:]
        df_val_c = df_test_c.iloc[val_c_idx,:] 
        if 'pad' in exp:
            df_val = self.pad(df_val)
            df_val_c = self.pad(df_val_c)

        os.makedirs(
            self.result_path+f'./test_{nb_test}/{exp}/data', 
            exist_ok=True)
        df_val.to_csv(
            self.result_path+f'./test_{nb_test}/{exp}/data/val.csv'
            )
        df_val_c.to_csv(
            self.result_path+f'./test_{nb_test}/{exp}/data/val_c.csv'
            )
        print('val split done')

    
    def train_sets(self, nb_test, exp):
        path = self.result_path+f'./test_{nb_test}/{exp}/data/val_c.csv'
        df_val_c = pd.read_csv(path, index_col = 0, header = [0,1])
        df_val_c_prop = df_val_c[('prop_id','0')]
        for frac in self.train_fracs:
            if frac == 100:
                train_idx = np.arange(len(df_val_c.index))
            else:
                train_idx, _ = train_test_split(
                    np.arange(len(df_val_c.index)),
                    train_size = frac/100, 
                    random_state = self.random_state,
                    shuffle = True,
                    stratify = df_val_c_prop
                    )
            df_train = df_val_c.iloc[train_idx,:]
            
            # Reset index for proper scaling in the setup of datamodule.
            if exp == 'only_p1_single': 
                # no padding, single-task loss
                df_train = self.only_p1(df_train).reset_index(drop=True)
            elif exp == 'aux_p_pad':
                # all auxiliary properties, padding, mtl loss
                df_train = self.aux_p(df_train).reset_index(drop=True)
            elif exp == 'aux_p_random_pad':
                # all auxiliary properties with random values, padding,
                # mtl loss
                df_train = self.aux_p_random(df_train).reset_index(drop=True)
            else:
                ValueError('Invalid exp')
            df_train.to_csv(
                self.result_path\
                    +f'./test_{nb_test}/{exp}/data/train_{frac}.csv'
                )
    

    def gen_train_val_test_sets(self, ids):
        for exp in ids:
            self.test_splits(exp)
            for i in range(self.n_folds):
                self.val_splits(i, exp)
                self.train_sets(i, exp)