"""
Methods for data evaluation
"""

from typing import List

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os


###
# Samples
###
class Samples():
    """
    Class to evaluate the number of data samples.
    """
    def __init__(
            self,
            ident: str,    # identifier
            props: List[str],
            exps: List[str],
            n_folds: int = 5,
            fracs: List[int] = [1,2,4,8,16,32,64,100],
            seed: int = 42 
            ):
        self.ident = ident
        self.props = props
        self.exps = exps
        self.n_folds = n_folds
        self.fracs = fracs
        self.seed = seed
    

    def samples_train(self, prop: str, exp: str) -> pd.DataFrame:
        n_samples = []
        for fold in range(self.n_folds):
            samples_per_test = []
            for frac in self.fracs:
                path = f'results/{self.ident}/{prop}/seed{self.seed}/test_{fold}/\
{exp}/data/train_{frac}.csv'
                df = pd.read_csv(path, index_col = 0, header = [0,1])
                samples_per_test.append(df.shape[0])
            n_samples.append(samples_per_test)
        return pd.DataFrame(
            n_samples, 
            index = range(self.n_folds), 
            columns=self.fracs
            )


    def samples_test(self, prop: str, exp: str) -> pd.DataFrame:
        n_samples = []
        for fold in range(self.n_folds):
            samples_per_test = []
            path = f'results/{self.ident}/{prop}/seed{self.seed}/test_{fold}/\
{exp}/data/test.csv'
            df = pd.read_csv(path, index_col = 0, header = [0,1])
            samples_per_test.append(df.shape[0])
            n_samples.append(samples_per_test)
        return pd.DataFrame(n_samples, index = range(self.n_folds))


    def samples_val(self, prop: str, exp: str) -> pd.DataFrame:
        n_samples = []
        for fold in range(self.n_folds):
            samples_per_test = []
            path = f'results/{self.ident}/{prop}/seed{self.seed}/test_{fold}/\
{exp}/data/val.csv'
            df = pd.read_csv(path, index_col = 0, header = [0,1])
            samples_per_test.append(df.shape[0])
            n_samples.append(samples_per_test)
        return pd.DataFrame(n_samples, index = range(self.n_folds))


    def samples_all(self) -> None:
        """
        Generate table showing the number of data samples in train, 
        validation, and test sets.
        """
        data_coll = {}
        for prop in self.props:
            for exp in self.exps:
                data = {}
                data['prop'] = prop
                data['exp'] = exp
                df_test = self.samples_test(prop, exp)
                data['test'] = round(df_test.mean(axis=0).loc[0])
                
                df_val = self.samples_val(prop, exp)
                data['val'] = round(df_val.mean(axis=0).loc[0])
                
                df_train = self.samples_train(prop, exp)
                for i in self.fracs:
                    data[f'train_{i}'] = round(df_train.mean(axis=0).loc[i])
                
                data_coll[prop + '_' + exp] = data
                    
        df = pd.DataFrame.from_dict(
            data_coll, 
            orient = 'index'
            ).reset_index(drop = True)

        path = f'results/{self.ident}/eval/'
        os.makedirs(path, exist_ok = True)
        df.to_csv(path+f'samples.csv', index = False)


    def stats(self) -> None:
        """
        Generate table showing descriptive statistics of the data.
        """
        props = ['Tg', 'YoungsModulus293K', 'RefractiveIndex', 'Density293K']
        viscs = ['LogViscosity773K', 'LogViscosity873K', 'LogViscosity973K', 
                 'LogViscosity1073K', 'LogViscosity1173K', 'LogViscosity1273K', 
                 'LogViscosity1373K', 'LogViscosity1473K', 'LogViscosity1573K', 
                 'LogViscosity1673K', 'LogViscosity1773K', 'LogViscosity1873K', 
                 'LogViscosity2073K', 'LogViscosity2273K', 'LogViscosity2473K']
        props = props + viscs

        stat = {}
        for prop in props:
            df = pd.read_csv(
                f'datasets/{prop}/all.csv', 
                index_col = 0, 
                header = [0,1]
                )
            stat[prop] = {
                'samples': int(df.shape[0]), 
                'elements': int(df.elements.shape[1]), 
                'min': round(df.prop_val.min().iloc[0], 3), 
                'mean': round(df.prop_val.mean().iloc[0],3), 
                'max': round(df.prop_val.max().iloc[0],3)
                } 
            df = pd.read_csv(
                f'datasets/LogViscosity/all.csv', 
                index_col = 0, 
                header = [0,1]
                )
            
        stat['LogViscosity'] = {
            'samples': int(df.shape[0]), 
            'elements': int(df.elements.shape[1]), 
            'min': round(df.prop_val.min().iloc[0], 3), 
            'mean': round(df.prop_val.mean().iloc[0],3), 
            'max': round(df.prop_val.max().iloc[0],3)
            } 

        df_stat = pd.DataFrame.from_dict(stat, orient = 'index')
        df_stat.to_csv(f'results/{self.ident}/eval/stats.csv')




###
# Error metrics
###
class Errors():
    """
    Class to evaluate errors.
    """
    def __init__(            
            self,
            ident: str,
            props: List[str],
            exps: List[str],
            n_folds: int = 5,
            fracs: List[int] = [1,2,4,8,16,32,64,100],
            seeds: List[int] = [1, 42, 100]
            ):
        self.ident = ident
        self.props = props
        self.exps = exps
        self.n_folds = n_folds
        self.fracs = fracs
        self.ident = ident
        self.seeds = seeds
        self.rmse = lambda df: metrics.root_mean_squared_error(
            df.y.values, 
            df.yhat.values
            )
        self.mae = lambda df: metrics.mean_absolute_error(
            df.y.values, 
            df.yhat.values
            )
        self.r2 = lambda df: metrics.r2_score(
            df.y.values, 
            df.yhat.values
            )
        self.errors_all = {}
        self.linear_mae_models = {}
        self.path = f'results/{self.ident}/eval/'
        os.makedirs(self.path, exist_ok = True)

    def error(error_func, path: str) -> float:
        df = pd.read_csv(path, index_col = 0, header = [0,1])
        err = error_func(df)
        return err
    

    def compute_errors(self, prop: str, exp: str, round_prec: int) -> dict:
        """
        Compute MAEs, RMSEs, and R2s.
        """
        errors_dict = {}
        for frac in self.fracs:
            errors_dict[frac] = {}
            errors_dict[frac]['mae'] = {}
            errors_dict[frac]['rmse'] = {}
            errors_dict[frac]['r2'] = {}
            errors_dict[frac]['mae_stat'] = []
            errors_dict[frac]['rmse_stat'] = []
            errors_dict[frac]['r2_stat'] = []

            for seed in self.seeds:
                errors_dict[frac]['mae'][f'seed{seed}'] = []
                errors_dict[frac]['rmse'][f'seed{seed}']  = []
                errors_dict[frac]['r2'][f'seed{seed}']  = []
                for fold in range(self.n_folds):
                    path = f'results/{self.ident}/{prop}/seed{seed}/\
test_{fold}/{exp}/results/test_{frac}.csv'
                    df = pd.read_csv(path, index_col = 0, header = 0)
                    errors_dict[frac]['mae'][f'seed{seed}'].append(self.mae(df))
                    errors_dict[frac]['rmse'][f'seed{seed}'].append(self.rmse(df))
                    errors_dict[frac]['r2'][f'seed{seed}'].append(self.r2(df))

            maes = [v for lst in errors_dict[frac]['mae'].values() for v in lst]
            rmses = [v for lst in errors_dict[frac]['rmse'].values() for v in lst]
            r2s = [v for lst in errors_dict[frac]['r2'].values() for v in lst]
            errors_dict[frac]['mae_stat'] = (
                round(np.array(maes).mean(),round_prec), 
                round(np.array(maes).std(),round_prec)
                )
            errors_dict[frac]['rmse_stat'] = (
                round(np.array(rmses).mean(),round_prec), 
                round(np.array(rmses).std(),round_prec)
                )
            errors_dict[frac]['r2_stat'] = (
                round(np.array(r2s).mean(),round_prec), 
                round(np.array(r2s).std(),round_prec)
                )
        return errors_dict
    

    def save_errors(self) -> None:
        """
        Generate tables showing MAEs, RMSEs, and R2s.
        """            
        for prop in self.props: 
            self.errors_all[prop] = {}
            for exp in self.exps:
                if 'LogViscosity' in prop:
                    round_prec = 3
                elif 'YoungsModulus' in prop:
                    round_prec = 2
                self.errors_all[prop][exp] = self.compute_errors(
                    prop, 
                    exp, 
                    round_prec = round_prec
                    )                         
        # All seeds
        for error_name in ['mae', 'rmse', 'r2']:
            errors_lst = []
            for prop in self.props:
                for exp in self.exps:
                    errors = {'prop': prop, 'exp': exp}
                    for frac in self.fracs:                           
                        err = f'{
                            self.errors_all[prop][exp][frac][error_name+'_stat'][0]
                            }({self.errors_all[prop][exp][frac][error_name+'_stat'][1]})'
                        # Compute relative errors
                        if 'aux' in exp:
                            if 'YoungsModulus' in prop:
                                err_rel = round(
                                    100*(-1 + self.errors_all[prop]\
                                            [exp][frac][error_name+'_stat'][0]/
                                    self.errors_all['YoungsModulus293K_Density293K']\
                                        ['only_p1_single'][frac][error_name+'_stat'][0]), 
                                    2
                                    )
                            else:
                                err_rel = round(
                                    100*(-1 + self.errors_all[prop]\
                                            [exp][frac][error_name+'_stat'][0]/
                                    self.errors_all[prop]\
                                        ['only_p1_single'][frac][error_name+'_stat'][0]), 
                                    2
                                    )

                        else: 
                            err_rel = 0.
                        errors[frac] = err
                        errors[f'{frac}_rel'] = err_rel
                    errors_lst.append(errors)          
            df = pd.DataFrame(errors_lst)
            df.to_csv(self.path+f'{error_name}s.csv', index = False)


    def linear_fit(self) -> None:
        """
        Generate table showing slopes of linear fits.
        """
        for prop in self.props:
            self.linear_mae_models[prop] = {}
            for exp in ['only_p1_single', 'aux_p_pad']:
                X = np.log10(np.array(self.fracs)).reshape(-1,1)
                y = np.log10(
                    np.array(
                        [self.errors_all[prop][exp][frac]['mae_stat'][0]\
                         for frac in self.fracs]
                         )
                    )
                reg = LinearRegression().fit(X, y)
                self.linear_mae_models[prop][exp] = round(reg.coef_[0], 2)
        df = pd.DataFrame.from_dict(
            self.linear_mae_models, 
            orient = 'index'
            ).reset_index()
        df.to_csv(self.path+f'linear_fit.csv', index = False)

    
    def plot_YM(self) -> None:
        """
        Generate plot for Young's modulus as target property.
        """
        mae_test_YM1, rmse_test_YM1, r2_test_YM1 = [], [], []
        prop1 = 'YoungsModulus293K_Density293K'
        exps1 = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']
        labels1 = [r'$E$', r'$E$_$\rho$', r'$E$_$\rho$_random']
        colors1 = ['black', 'tab:green', 'tab:orange']
        for exp in exps1:
            mae_test_YM1.append(
                [(self.errors_all[prop1][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop1][exp][frac]['mae_stat'][1]) 
                 for frac in self.fracs]
                 )
            rmse_test_YM1.append(
                [(self.errors_all[prop1][exp][frac]['rmse_stat'][0], 
                  self.errors_all[prop1][exp][frac]['rmse_stat'][1]) 
                 for frac in self.fracs]
                 )
            r2_test_YM1.append(
                [(self.errors_all[prop1][exp][frac]['r2_stat'][0], 
                  self.errors_all[prop1][exp][frac]['r2_stat'][1]) 
                 for frac in self.fracs]
                 )

        mae_test_YM2, rmse_test_YM2, r2_test_YM2 = [], [], []
        prop2 = 'YoungsModulus293K_Density293K_RefractiveIndex'
        exps2 = ['aux_p_pad', 'aux_p_random_pad']
        labels2 = [r'$E$_$\rho$_$n_D$', r'$E$_$\rho$_$n_D$_random']
        colors2 = ['tab:red', 'tab:blue']
        samples = pd.read_csv(
            f'results/{self.ident}/eval/samples.csv'
            ).iloc[0,4:].values

        for exp in exps2:
            mae_test_YM2.append(
                [(self.errors_all[prop2][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop2][exp][frac]['mae_stat'][1])\
                    for frac in self.fracs]
                  )
            rmse_test_YM2.append(
                [(self.errors_all[prop2][exp][frac]['rmse_stat'][0], 
                  self.errors_all[prop2][exp][frac]['rmse_stat'][1])\
                    for frac in self.fracs]
                  )
            r2_test_YM2.append(
                [(self.errors_all[prop2][exp][frac]['r2_stat'][0], 
                  self.errors_all[prop2][exp][frac]['r2_stat'][1])\
                    for frac in self.fracs]
                  )

        lw = 2.5
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(
            3, 
            1, 
            layout='constrained', 
            figsize = (15,21), 
            sharex = True
            )

        for j in range(len(exps1)):
            ax[0].errorbar(
                self.fracs, 
                [mae_test_YM1[j][i][0] for i in range(len(self.fracs))], 
                [mae_test_YM1[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels1[j]}', 
                capsize = 4.0, 
                marker='x', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[1].errorbar(
                self.fracs, 
                [rmse_test_YM1[j][i][0] for i in range(len(self.fracs))], 
                [rmse_test_YM1[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '-', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[2].errorbar(
                self.fracs, 
                [r2_test_YM1[j][i][0] for i in range(len(self.fracs))], 
                [r2_test_YM1[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '-', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )

        for j in range(len(exps2)):
            ax[0].errorbar(
                self.fracs, 
                [mae_test_YM2[j][i][0] for i in range(len(self.fracs))], 
                [mae_test_YM2[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels2[j]}', 
                capsize = 4.0, 
                marker='x', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[1].errorbar(
                self.fracs, 
                [rmse_test_YM2[j][i][0] for i in range(len(self.fracs))], 
                [rmse_test_YM2[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '-', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[2].errorbar(
                self.fracs, 
                [r2_test_YM2[j][i][0] for i in range(len(self.fracs))], 
                [r2_test_YM2[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '-', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
        
        # Linear fit
        exponent1 = self.linear_mae_models[prop1]['only_p1_single']
        exponent2 = self.linear_mae_models[prop2]['aux_p_pad']
        ax[0].plot(
            [1, 16], 
            [7, 7*16**exponent1], 
            linestyle = '-.', 
            color = 'black', 
            label = fr"$m^{{{exponent1:,.2f}}}$", 
            linewidth = lw
            )
        ax[0].plot(
            [1, 16], 
            [7, 7*16**exponent2], 
            linestyle = '-.', 
            color = 'tab:red',
            label = fr"$m^{{{exponent2:,.2f}}}$", 
            linewidth = lw
            )

        # Labels and legend
        ax[0].legend()
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_yticks(np.arange(3,15), np.arange(3,15))
        ax[0].set_xticks(self.fracs,self.fracs)
        ax[0].set_ylabel('MAE [GPa]')
        secax1 = ax[0].secondary_xaxis(
            'top', 
            functions=(lambda x: x, lambda x: x)
            )
        secax1.set_xlabel(r'#Training samples $m$')
        secax1.set_xticks(self.fracs,self.fracs,rotation=45)
        secax1.set_xticklabels(
            [f'{samples[i]}' for i in range(len(secax1.get_xticks()))]
            )
        handles, labels = ax[0].get_legend_handles_labels()
        order = [2,3,4,5,6,0,1]
        ax[0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order]
            ).get_frame().set_edgecolor('black')
        ax[0].grid()

        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_xticks(self.fracs,self.fracs)
        ax[1].set_yticks(np.arange(6,20), np.arange(6,20))
        ax[1].set_ylabel('RMSE [GPa]')
        ax[1].grid()

        ax[2].set_xscale('log')
        ax[2].set_xticks(self.fracs,self.fracs)
        ax[2].set_yticks(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
        ax[2].set_xlabel('Training set fractions [%]')
        ax[2].set_ylabel('$R^2$')
        ax[2].grid()

        plt.savefig(self.path + 'YM_Density_RI.png', dpi = 600)


    def plot_YM_MAE(self) -> None:
        """
        Generate MAE plot for Young's modulus as target property.
        """
        mae_test_YM1 = []
        prop1 = 'YoungsModulus293K_Density293K'
        exps1 = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']
        labels1 = [r'$E$', r'$E$_$\rho$', r'$E$_$\rho$_random']
        colors1 = ['black', 'tab:green', 'tab:orange']
        for exp in exps1:
            mae_test_YM1.append(
                [(self.errors_all[prop1][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop1][exp][frac]['mae_stat'][1]) 
                 for frac in self.fracs]
                 )

        mae_test_YM2 = []
        prop2 = 'YoungsModulus293K_Density293K_RefractiveIndex'
        exps2 = ['aux_p_pad', 'aux_p_random_pad']
        labels2 = [r'$E$_$\rho$_$n_D$', r'$E$_$\rho$_$n_D$_random']
        colors2 = ['tab:red', 'tab:blue']
        samples = pd.read_csv(
            f'results/{self.ident}/eval/samples.csv'
            ).iloc[0,4:].values

        for exp in exps2:
            mae_test_YM2.append(
                [(self.errors_all[prop2][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop2][exp][frac]['mae_stat'][1])\
                    for frac in self.fracs]
                  )

        lw = 2.5
        plt.rcParams.update({'font.size': 25})
        fig, ax = plt.subplots(
            1, 
            1, 
            layout='constrained', 
            figsize = (15,10), 
            sharex = True
            )

        for j in range(len(exps1)):
            ax.errorbar(
                self.fracs, 
                [mae_test_YM1[j][i][0] for i in range(len(self.fracs))], 
                [mae_test_YM1[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels1[j]}', 
                capsize = 5.0, 
                marker='x', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )
        for j in range(len(exps2)):
            ax.errorbar(
                self.fracs, 
                [mae_test_YM2[j][i][0] for i in range(len(self.fracs))], 
                [mae_test_YM2[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels2[j]}', 
                capsize = 5.0, 
                marker='x', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
        
        # Linear fit
        exponent1 = self.linear_mae_models[prop1]['only_p1_single']
        exponent2 = self.linear_mae_models[prop2]['aux_p_pad']
        ax.plot(
            [1, 16], 
            [7, 7*16**exponent1], 
            linestyle = '-.', 
            color = 'black', 
            label = fr"$m^{{{exponent1:,.2f}}}$", 
            linewidth = lw
            )
        ax.plot(
            [1, 16], 
            [7, 7*16**exponent2], 
            linestyle = '-.', 
            color = 'tab:red',
            label = fr"$m^{{{exponent2:,.2f}}}$", 
            linewidth = lw
            )
        
        # Labels and legend
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_yticks(np.arange(3,15), np.arange(3,15))
        ax.set_xticks(self.fracs,self.fracs)
        ax.set_xlabel('Training fractions [%]')
        ax.set_ylabel('MAE [GPa]')
        secax1 = ax.secondary_xaxis(
            'top', 
            functions=(lambda x: x, lambda x: x)
            )
        secax1.set_xlabel(r'#Training samples $m$')
        secax1.set_xticks(self.fracs,self.fracs,rotation=45)
        secax1.set_xticklabels(
            [f'{samples[i]}' for i in range(len(secax1.get_xticks()))]
            )
        handles, labels = ax.get_legend_handles_labels()
        order = [2,3,4,5,6,0,1]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order]
            ).get_frame().set_edgecolor('black')
        ax.grid()

        plt.savefig(self.path + 'YM_Density_RI_MAE.png', dpi = 600)


    def plot_LogViscosity(self) -> None:
        """
        Generate plot for LogViscosity as target property.
        """
        mae_test1, rmse_test1, r2_test1 = [], [], []
        prop1 = 'LogViscosity_Tg'
        exps1 = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']
        labels1 = [r'$\log_{10}\eta_{T_i}$', 
                   r'$\log_{10}\eta_{T_i}$_$T_g$', 
                   r'$\log_{10}\eta_{T_i}$_$T_g$_random'
                   ]
        colors1 = ['black', 'tab:green', 'tab:orange']
        for exp in exps1:
            mae_test1.append(
                [(self.errors_all[prop1][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop1][exp][frac]['mae_stat'][1])\
                    for frac in self.fracs]
                  )
            rmse_test1.append(
                [(self.errors_all[prop1][exp][frac]['rmse_stat'][0], 
                  self.errors_all[prop1][exp][frac]['rmse_stat'][1])\
                    for frac in self.fracs]
                  )
            r2_test1.append(
                [(self.errors_all[prop1][exp][frac]['r2_stat'][0], 
                  self.errors_all[prop1][exp][frac]['r2_stat'][1])\
                    for frac in self.fracs]
                  )

        mae_test2, rmse_test2, r2_test2 = [], [], []
        prop2 = 'LogViscosity_Tg_t'
        exps2 = ['only_p1_single', 'aux_p_pad', 'aux_p_random_pad']
        labels2 = [r'$\log_{10}\eta_{t}$', 
                   r'$\log_{10}\eta_{t}$_$T_g$', 
                   r'$\log_{10}\eta_{t}$_$T_g$_random'
                   ]
        colors2 = ['tab:purple', 'tab:red', 'tab:blue']

        samples = pd.read_csv(
            f'results/{self.ident}/eval/samples.csv'
            ).iloc[6,4:].values

        for exp in exps2:
            mae_test2.append(
                [(self.errors_all[prop2][exp][frac]['mae_stat'][0], 
                  self.errors_all[prop2][exp][frac]['mae_stat'][1]) 
                  for frac in self.fracs]
                  )
            rmse_test2.append(
                [(self.errors_all[prop2][exp][frac]['rmse_stat'][0], 
                  self.errors_all[prop2][exp][frac]['rmse_stat'][1]) 
                  for frac in self.fracs]
                  )
            r2_test2.append(
                [(self.errors_all[prop2][exp][frac]['r2_stat'][0], 
                  self.errors_all[prop2][exp][frac]['r2_stat'][1]) 
                  for frac in self.fracs]
                  )

        lw = 2.5
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(
            3, 
            1, 
            layout='constrained', 
            figsize = (15,21), 
            sharex = True
            )
        for j in range(len(exps1)):
            ax[0].errorbar(
                self.fracs, 
                [mae_test1[j][i][0] for i in range(len(self.fracs))], 
                [mae_test1[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels1[j]}', 
                capsize = 4.0, 
                marker='x', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[1].errorbar(
                self.fracs, 
                [rmse_test1[j][i][0] for i in range(len(self.fracs))], 
                [rmse_test1[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[2].errorbar(
                self.fracs, 
                [r2_test1[j][i][0] for i in range(len(self.fracs))], 
                [r2_test1[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                color = colors1[j], 
                linewidth = lw, 
                ms = 8
                )

        for j in range(len(exps2)):
            ax[0].errorbar(
                self.fracs, 
                [mae_test2[j][i][0] for i in range(len(self.fracs))], 
                [mae_test2[j][i][1] for i in range(len(self.fracs))], 
                label = f'{labels2[j]}', 
                capsize = 4.0, 
                marker='x', 
                linestyle = '--', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[1].errorbar(
                self.fracs, 
                [rmse_test2[j][i][0] for i in range(len(self.fracs))], 
                [rmse_test2[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '--', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
            ax[2].errorbar(
                self.fracs, 
                [r2_test2[j][i][0] for i in range(len(self.fracs))], 
                [r2_test2[j][i][1] for i in range(len(self.fracs))], 
                capsize = 4.0, 
                marker='x', 
                linestyle = '--', 
                color = colors2[j], 
                linewidth = lw, 
                ms = 8
                )
        
        # Linear fit
        exponent1 = self.linear_mae_models[prop1]['only_p1_single']
        exponent2 = self.linear_mae_models[prop2]['aux_p_pad']
        ax[0].plot(
            [1, 16], 
            [0.4, 0.4*16**exponent1], 
            linestyle = '-.', 
            color = 'tab:purple', 
            label = fr"$m^{{{exponent1:,.2f}}}$", 
            linewidth = lw
            )
        ax[0].plot(
            [1, 16], 
            [0.4, 0.4*16**exponent2], 
            linestyle = '-.', 
            color = 'tab:red',
            label = fr"$m^{{{exponent2:,.2f}}}$", 
            linewidth = lw
            )
        
        # Labels and legend
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xticks(self.fracs,self.fracs)
        ax[0].set_yticks(np.arange(2,10)/10, np.arange(2,10)/10)
        ax[0].set_ylabel('MAE [$log_{10}$Pa s]')
        secax1 = ax[0].secondary_xaxis(
            'top', 
            functions=(lambda x: x, lambda x: x)
            )
        secax1.set_xlabel(r'#Training samples $m$')
        secax1.set_xticks(self.fracs,self.fracs,rotation=45)
        secax1.set_xticklabels(
            [f'{samples[i]}' for i in range(len(secax1.get_xticks()))]
            )
        handles, labels = ax[0].get_legend_handles_labels()
        order = [2,3,4,5,6,7,0,1]
        ax[0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order]
            ).get_frame().set_edgecolor('black')
        ax[0].grid()

        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xticks(self.fracs,self.fracs)
        ax[1].set_yticks(np.arange(3,15)/10, np.arange(3,15)/10)
        ax[1].set_ylabel('RMSE [$log_{10}$Pa s]')
        ax[1].grid()

        ax[2].set_xscale('log')
        ax[2].set_xticks(self.fracs,self.fracs)
        ax[2].set_yticks(
            [0.6, 0.7, 0.8, 0.9, 1.0], 
            [0.6, 0.7, 0.8, 0.9, 1.0]
            )
        ax[2].set_xlabel('Training set fractions [%]')
        ax[2].set_ylabel('$R^2$')
        ax[2].grid()

        plt.savefig(self.path + 'LogViscosity_Tg.png', dpi=600)