"""
DataSet and DataModule utilities
"""


from typing import List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
import random
import os
from scripts import params


###########
# Ensure reproducibility
###########
def seed_everything(seed=params.SEED):
    """Set seed for reproducibility."""
    pl.seed_everything(seed, workers=False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


############
# Dataset and Dataloader
############
class GlassDataset(Dataset):
    def __init__(
            self,
            X: np.array,
            y: np.array,
            n_elems: int,
            prop_names: List[str],
            with_t: bool,
            t_range: Optional[List[float]] = None):
        """
        :param X: 2D array with atomic symbols, atomic fractions,
            properties, and t-value horizontally concatenated
            of shape (n_samples, 2*:obj:`n_elements` + 2)
        :param y: 2D array with property values of shape (n_samples, 1)
        :param n_elems: Number of elements.
        :param prop_names: List of property names.
        :param with_t: Boolean indicating t-parametrization.
        :param t_range: Range of the t-parameter.
        """
        self.X = X
        self.y = y
        self.n_elems = n_elems
        self.with_t = with_t
        self.t_range = t_range
        self.elements = params.ELEMENTS

        self.le_elements = preprocessing.LabelEncoder()
        self.le_elements.fit(self.elements)
        self.le_props = preprocessing.LabelEncoder()
        self.le_props.fit(prop_names)


    def __len__(self) -> int:
        return self.X.shape[0]  # number of samples
    

    def element2idx(self, x: np.array) -> np.array:
        return self.le_elements.transform(x)
    

    def prop2idx(self, x: np.array) -> np.array:
        return self.le_props.transform(x)


    def scale_t(self, t: np.array) -> np.array:
        if (t < self.t_range[0]) or (t > self.t_range[1]):
            raise ValueError(f"t is out of range!\
It has to be in {self.t_range}.")
        return (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0])
    

    def __getitem__(self, idx):
        elems = torch.tensor(
            self.element2idx(self.X[idx, :self.n_elems]), 
            dtype = torch.int32
            )
        fracs = torch.tensor(
            self.X[idx, self.n_elems:2*self.n_elems].astype(float),
            dtype = torch.float32
            )
        y = torch.tensor(self.y[idx], dtype = torch.float32)
        if self.with_t:
            prop = torch.tensor(
                self.prop2idx([self.X[idx, -2]]), 
                dtype = torch.int32
                )
            t = torch.tensor(
                self.scale_t(np.array([self.X[idx, -1]])), 
                dtype = torch.float32
                )        
            return (elems, fracs, prop, t), y
        else:  
            prop = torch.tensor(
                self.prop2idx([self.X[idx, -1]]), 
                dtype = torch.int32
                )
            return (elems, fracs, prop), y



class GlassDataModule(pl.LightningDataModule):
    def __init__(
            self,
            prop_groups: List[List[str]],
            scalers: List,
            with_t: bool = False,
            t_range: Optional[List[float]] = None,
            train_batch = 32,
            val_batch = 32,
            train_path: Optional[str] = None,
            val_path: Optional[str] = None,
            test_path: Optional[str] = None,
            predict_path: Optional[str] = None
            ):
        """
        :param prop_groups: List of property groups. 
            Example: [['LogViscosity673K', 'LogViscosity673K'], ['Tg']]
        :param scalers: List of scalers for each property group.
        :param with_t: Boolean indicating whether the properties are 
            parametrized by t or not.
        :param t_range: Range of t-parameter or None. 
        :param train_batch: Batch size for training.
        :param val_batch: Batch size for validation.
        :param train_path: Path to save training set to.
        :param val_path: Path to save validation set to.
        :param test_path: Path to save test set to.
        :param predict_path: Path to save dataset for extra predictions to.
        """
        super().__init__()
 
        self.scalers = scalers
        self.prop_groups = prop_groups
        self.prop_names = list(
            itertools.chain.from_iterable(self.prop_groups)
            )
        self.with_t = with_t
        self.t_range = t_range
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.g = torch.Generator()
        self.g.manual_seed(params.SEED)
        

    def find_p(self, row: str, names: List[str]):
        if row in names:
            return row
        else:
            return np.nan
        

    def scale(
            self, 
            df: pd.DataFrame, 
            fit_scaler: bool = False
            ) -> np.array:
        """
        Scale each property in :obj:`df` individually.
        """
        y_name = df[('prop_name','0')]
        y = df[('prop_val','0')]
        for i, name in enumerate(self.prop_groups):
            idx = y_name.apply(
                lambda row: self.find_p(row, name)
                ).dropna().index
            y_val = y.loc[idx].values.reshape(-1,1)
            if len(idx)>0:
                if fit_scaler:
                    self.scalers[i].fit(y_val)
                y.loc[idx] = self.scalers[i].transform(y_val).ravel()
        return y.values.reshape(-1,1)


    def setup(self, stage: str):
        if stage == 'fit':
            # Training set
            df_train = pd.read_csv(
                self.train_path,
                index_col = 0,
                header = [0,1]
                )
            if self.with_t:
                X_train = df_train[
                    ['elements', 'at_fracs', 'prop_name', 't']
                    ].values
            else:
                X_train = df_train[
                    ['elements', 'at_fracs', 'prop_name']
                    ].values
                
            y_train_scaled = self.scale(df_train, fit_scaler = True)
            n_elems = df_train.elements.shape[1]
            self.ds_train = GlassDataset(
                X_train, 
                y_train_scaled, 
                n_elems, 
                self.prop_names,
                self.with_t, 
                self.t_range
                )
            print('train setup done')

            # Validation set
            df_val = pd.read_csv(
                self.val_path,
                index_col = 0,
                header = [0,1]
                )
            if self.with_t:
                X_val = df_val[
                    ['elements', 'at_fracs', 'prop_name', 't']
                    ].values
            else:
                X_val = df_val[['elements', 'at_fracs', 'prop_name']].values
            y_val = df_val[('prop_val','0')].values.reshape(-1,1)
            y_val_scaled = self.scalers[0].transform(y_val)
            self.ds_val = GlassDataset(
                X_val, 
                y_val_scaled, 
                n_elems, 
                self.prop_names,
                self.with_t,
                self.t_range
                )
            print('val setup done')

        if stage == 'test':
            # Test set
            df_test = pd.read_csv(
                self.test_path,
                index_col = 0,
                header = [0,1]
                )
            if self.with_t:
                X_test = df_test[
                    ['elements', 'at_fracs', 'prop_name', 't']
                    ].values
            else:
                X_test = df_test[['elements', 'at_fracs', 'prop_name']].values
            y_test = df_test[('prop_val','0')].values.reshape(-1,1)
            y_test_scaled = self.scalers[0].transform(y_test)
            n_elems = df_test.elements.shape[1]
            self.ds_test = GlassDataset(
                X_test, 
                y_test_scaled, 
                n_elems, 
                self.prop_names,
                self.with_t,
                self.t_range
                )
            
        if stage == 'predict':
            # Set for inference
            df_predict = pd.read_csv(
                self.predict_path,
                index_col = 0,
                header = [0,1]
                )
            if self.with_t:
                X_predict = df_predict[
                    ['elements', 'at_fracs', 'prop_name', 't']
                    ].values
            else:
                X_predict = df_predict[
                    ['elements', 'at_fracs', 'prop_name']
                    ].values
            y_predict = df_predict[('prop_val','0')].values.reshape(-1,1)
            y_predict_scaled = self.scalers[0].transform(y_predict)
            n_elems = df_predict.elements.shape[1]
            self.ds_predict = GlassDataset(
                X_predict, 
                y_predict_scaled, 
                n_elems, 
                self.prop_names,
                self.with_t,
                self.t_range
                )


    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            shuffle=True,
            batch_size=self.train_batch,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=self.g,
            )


    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            shuffle=False,
            batch_size=self.val_batch,
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=self.g,
            )


    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=len(self.ds_test),
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=self.g,
            )
    

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            shuffle=False,
            batch_size=len(self.ds_predict),
            num_workers=8,
            worker_init_fn=seed_worker,
            generator=self.g,
            )