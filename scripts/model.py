"""
Collection of models
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CyclicLR
import pandas as pd
import numpy as np

from scripts import params
from scripts.optims import Lookahead


class ElementEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_all_elems: int
            ):
        """
        :param d_model: Latent model dimension.
        :param n_all_elem: Number of all elements that a glass consists of
            (including padding element 'X').
        """
        super(ElementEmbedding, self).__init__()

        # Padded element 'X' gets mapped to zero-th embedding row,
        # which is the zero vector
        self.embedding = nn.Embedding(
            n_all_elems, 
            d_model, 
            padding_idx = 0
            )


    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of size (batch_size, max_n_elem)
            with element encodings.
        
        :return: Tensor of size (batch_size, max_n_elem, d_model) 
            with element embeddings.
        """
        return self.embedding(x)
    



class PropertyEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int, 
            n_all_props: int
            ):
        super(PropertyEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_all_props, d_model)


    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of size (batch_size, 1) with property
            encodings.

        :return: Tensor of size (batch_size, 1, d_model) with
            property embeddings.
        """
        return self.embedding(x)




class FractionalEncoder(nn.Module):
    """
    Encode fractional amounts using a "fractional encoding";
    Adapted from https://github.com/anthony-wang/CrabNet/blob/master/crabnet/kingcrab.py 
    (see https://doi.org/10.1038/s41524-021-00545-1)
    """
    def __init__(
            self,
            d_model: int,    # Should be even!
            resolution: int=10000,
            log10: bool=False
            ):
        super().__init__()
        # Embedding of length d_model//2 to later 
        # concatenate non-log and log-embedding
        self.d_model = d_model//2
        # Equivalent to length of input sequence in NLP tasks
        self.resolution = resolution
        self.log10 = log10

        x = torch.linspace(
            0, 
            self.resolution-1,
            self.resolution,
            requires_grad=False
            ).view(self.resolution, 1)
        fraction = torch.linspace(
            0, 
            self.d_model-1,
            self.d_model,
            requires_grad=False
            ).view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(
            x/torch.pow(self.resolution, fraction[:, 0::2]/self.d_model)
            )
        pe[:, 1::2] = torch.cos(
            x/torch.pow(self.resolution, (fraction[:, 1::2]-1)/self.d_model)
            )
        # Zero vector at the bottom for encoding nan-values
        pe = torch.vstack(
            (pe, torch.zeros(1, self.d_model, requires_grad=False))
            )
        pe = self.register_buffer('pe', pe)


    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: A tensor of size (batch, L)
        :return: A tensor of size (batch, L, d_model//2)
        """
        x = x.clone()
        if self.log10:
            # Quadratic scaling on log-scale: 
            # Smaller values get weighted higher
            x = 0.04 * torch.log10(x)**2
            # Clamp x[x > 1] = 1
            x = torch.clamp(x, max=1) # Values < 10**(-5) get clamped
        # Clamp x[x < 1/self.resolution] = 1/self.resolution
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x*self.resolution)
        # NaN values get assigned index -1
        frac_idx[torch.isnan(frac_idx)] = 0
        frac_idx = frac_idx.to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out
    



class Encoder(nn.Module):
    def __init__(
            self,
            n_all_elems: int,
            n_all_props: int,
            d_model: int,
            n_attn: int,
            n_heads: int,
            with_t: bool = True,
            compute_device=None
            ):
        super().__init__()
        self.n_all_elems = n_all_elems
        self.n_all_props = n_all_props
        self.d_model = d_model
        self.n_attn = n_attn
        self.n_heads = n_heads
        self.with_t = with_t
        self.compute_device = compute_device
        self.elem_emb = ElementEmbedding(self.d_model, self.n_all_elems)
        self.prop_emb = PropertyEmbedding(self.d_model, self.n_all_props)
        self.frac_emb = FractionalEncoder(
            self.d_model, 
            resolution=10000, 
            log10=False
            )
        self.frac_log_emb = FractionalEncoder(
            self.d_model, 
            resolution=10000, 
            log10=True
            )
        if with_t:
            self.t_emb = FractionalEncoder(
                self.d_model*2, 
                resolution=10000, 
                log10=False
                )
            self.t_scaler = nn.parameter.Parameter(torch.tensor([1.]))

        self.elem_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.frac_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.frac_log_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.prop_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            nhead=self.n_heads,
            dim_feedforward=512, #2048,
            dropout=0.1,
            batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_attn
            )


    def forward(self, src) -> torch.tensor:
        """
        :return: Tensor of size (batch, max_n_elems + 1, d_model)
        """
        if self.with_t:
            (elems, fracs, prop, t) = src
        else:
            (elems, fracs, prop) = src
        
        elem_emb = self.elem_emb(elems)*2**self.elem_scaler
        frac_emb = torch.zeros_like(elem_emb)
        frac_log_emb = torch.zeros_like(elem_emb)
        frac_emb[:, :, :self.d_model//2]=self.frac_emb(fracs)\
                                            *2**self.frac_scaler
        frac_log_emb[:, :, self.d_model//2:] = self.frac_log_emb(fracs)\
                                                *2**self.frac_log_scaler
        prop_emb = self.prop_emb(prop)*2**self.prop_scaler
       
        src_elem = elem_emb + frac_emb + frac_log_emb
        if self.with_t:
            t_emb = self.t_emb(t)*2**self.t_scaler
            src_prop = prop_emb + t_emb
        else: 
            src_prop = prop_emb
        src = torch.cat((src_elem, src_prop), 1)

        # Mask to mark the padding areas in the keys that the attention layer 
        # should not attend to
        elem_mask = torch.where(fracs > 0, 1, 0)
        src_mask = torch.cat((elem_mask, torch.ones_like(prop)), dim=1) != 1
        y = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        return y




class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network as seen in Roost.
    https://doi.org/10.1038/s41467-020-19964-7
    Taken from 
    https://github.com/anthony-wang/CrabNet/blob/master/crabnet/kingcrab.py
    (see https://doi.org/10.1038/s41524-021-00545-1)
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layer_dims: List[int]
            ):
        super(ResidualNetwork, self).__init__()
        dims = [input_dim]+hidden_layer_dims
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1])for i in range(len(dims)-1)]
            )
        self.res_fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1], bias=False)
             if (dims[i] != dims[i+1])
             else nn.Identity()
             for i in range(len(dims)-1)]
            )
        self.acts = nn.ModuleList(
            [nn.LeakyReLU() for _ in range(len(dims)-1)]
            )
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea))+res_fc(fea)
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'




class GlassAttentionNet(pl.LightningModule):
    def __init__(
            self,
            n_props: int,
            d_model: int=128,
            n_attn: int=2,
            n_heads: int=4,
            with_t: bool=True,
            test_result_path: Optional[str] = None,
            predict_result_path: Optional[str] = None,
            loss: Optional[str] = None,
            ):
        super(GlassAttentionNet, self).__init__()

        self.save_hyperparameters()
        self.n_all_elems = len(params.ELEMENTS)
        self.n_all_props = n_props
        self.d_model = d_model
        self.n_attn = n_attn
        self.n_heads = n_heads
        self.with_t = with_t
        self.test_result_path = test_result_path
        self.predict_result_path = predict_result_path
        self.encoder = Encoder(
            self.n_all_elems,
            self.n_all_props,
            self.d_model,
            self.n_attn,
            self.n_heads,
            self.with_t
            )
        self.ResNet = ResidualNetwork(
            input_dim=self.d_model,
            output_dim=2,
            hidden_layer_dims=[128,64]
            )
        self.loss_weights = nn.Parameter(
            torch.ones(self.n_all_props, requires_grad=True)
            )
        self.loss_terms = nn.Parameter(
            torch.ones(len(self.loss_weights), requires_grad=False)
            )
        self.loss = loss
        self.weight_decay = 0.01


    def forward(self, x):
        xx = self.encoder(x)
        xxx = self.ResNet(xx)
        y = torch.einsum("ij,ij->ij", 
                         xxx[:, :, 0], 
                         nn.Sigmoid()(xxx[:, :, 1]))
        mask = x[2]    # encoded labels for properties
        return torch.mean(y, dim=1, keepdim=True).reshape(-1,1), mask
    

    def mtl_loss(self, loss_func, yhat, y, mask):
        """
        Compute the loss of multi-task learning with missing values.

        Reference:
          Liebel, L., and Körner, M. (2018). Auxiliary Tasks in Multi-task
          Learning (arXiv).

        :param mask: Tensor indicating which property the y-values refer to
        """

        weights = self.loss_weights
        nans = torch.tensor([np.nan], device = yhat.device)
        
        def strip_nans(x):
            return x[~x.isnan()]

        loss = 0.0
        valid_idx = []
        
        for i in range(len(weights)):
            yhat_s = strip_nans(
                torch.where(
                    (mask.flatten() == i).to(device=yhat.device), 
                    yhat.flatten(), 
                    nans
                    )
                )
            y_s = strip_nans(
                torch.where(
                    (mask.flatten() == i).to(device=y.device), 
                    y.flatten(), 
                    nans
                    )
                )
            if len(yhat_s) > 0:
                valid_idx.append(i)
                loss += (loss_func(yhat_s, y_s))*0.5/(weights[i]**2) \
                    + torch.log(1 + weights[i]**2)
                with torch.no_grad():
                    self.loss_terms[i] = loss_func(yhat_s, y_s)

        return loss
    

    def my_loss_func(self, yhat, y, mask):
        """
        Custom loss function. 
        Accordingly apply single task and multitask losses.
        """
        # Optimize MAE error
        if 'l1' in self.loss:
            if 'single' in self.loss:
                return F.l1_loss(yhat, y)
            elif 'multi' in self.loss:
                return self.mtl_loss(F.l1_loss, yhat, y, mask)
            else:
                raise ValueError('Invalid loss. Needs to contain "single" \
                           or "multi".')
        # Optionally optimize MSE error
        elif 'l2' in self.loss:
            if 'single' in self.loss:
                return F.mse_loss(yhat, y)
            elif 'multi' in self.loss:
                return self.mtl_loss(F.mse_loss, yhat, y, mask)
            else:
                raise ValueError('Invalid loss. Needs to contain "single" \
                           or "multi".')
        else:
            raise ValueError('Invalid loss. Needs to contain "l1" or "l2".')
        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat, mask = self.forward(x)
        loss = self.my_loss_func(yhat, y, mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat, mask = self.forward(x)
        loss = self.my_loss_func(yhat, y, mask)
        self.log("val_loss", loss, on_step = True, on_epoch = True)
        return {"loss": loss}
    

    def test_step(self, test_batch, batch_idx):
        scaler = self.trainer.datamodule.scalers[0]
        x, y = test_batch
        yhat, mask = self.forward(x)

        # Safe true and predicted results
        y_test = scaler.inverse_transform(y.detach().cpu().numpy())
        yhat_test = scaler.inverse_transform(yhat.detach().cpu().numpy())
        df_test = pd.DataFrame({'y': y_test.ravel(), 
                                'yhat': yhat_test.ravel()})
        df_test.to_csv(self.test_result_path)

        loss = self.my_loss_func(yhat, y, mask)
        self.log("test_loss", loss, on_step = True, on_epoch = True)

        return {"loss": loss}
    

    def predict_step(self, predict_batch, batch_idx):
        scaler = self.trainer.datamodule.scalers[0]
        x, y = predict_batch
        yhat, mask = self.forward(x)

        # Safe true and predicted results
        y_test = scaler.inverse_transform(y.detach().cpu().numpy())
        yhat_test = scaler.inverse_transform(yhat.detach().cpu().numpy())
        df_test = pd.DataFrame({'y': y_test.ravel(), 
                                'yhat': yhat_test.ravel()})
        df_test.to_csv(self.predict_result_path)


    def configure_optimizers(self):
        base_optim = torch.optim.Adam(params=self.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        lr_scheduler = CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=6e-3,
            cycle_momentum=False,
            step_size_up=len(self.trainer.datamodule.train_dataloader())
            )
        return {"optimizer": optimizer,
                "lr_scheduler" : {"scheduler" : lr_scheduler,
                                  "monitor": "val_loss_epoch"}
                }