# sc_mll.py

import torch
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from j_computer import batchJacobian_AD


class SensitivityAwareGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, train_sc, lambda_reg, subset_frac, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)
        self.train_sc = train_sc
        self.lambda_reg = lambda_reg
        self.subset_frac = subset_frac

    def get_sensitivity(self, x: Tensor) -> Tensor:
        if not x.requires_grad:
            x.requires_grad_(True)

        mean_x = self.posterior(x).mean
        
        # s_grads = batchJacobian_AD(mean_x.sum(), x, True, True)
        s_grads = torch.autograd.grad(
            outputs=mean_x.sum(),
            inputs=x,
            create_graph=True,
        )[0]

        return s_grads[..., -1].unsqueeze(-1)


class SensitivityAwareMLL(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model, lambda_reg=0.1, subset_frac=0.5, **kwargs):
        super().__init__(likelihood, model, **kwargs)
        # self.lambda_reg = 1e-4  #lambda_reg
        # print(self.lambda_reg)

    def forward(self, function_dist: MultivariateNormal, target: Tensor, *args, **kwargs) -> Tensor:
        # 1. Calculate the standard MLL
        self.lambda_reg = self.model.lambda_reg

        mll = super().forward(function_dist, target, *args, **kwargs)
        if self.lambda_reg > 0:
            train_x = self.model.train_inputs[0]
            train_sc = self.model.train_sc

            num_points = train_x.shape[0]
            subset_size = int(num_points * self.model.subset_frac)

            # --- Subsetting Logic ---
            # Create a random permutation of indices and select a subset
            perm = torch.randperm(num_points, device=train_x.device)
            idx_sub = perm[:subset_size]

            train_x_sub = train_x[idx_sub]
            train_sc_sub = train_sc[idx_sub]

        
            pred_sc = self.model.get_sensitivity(train_x)
            loss_sc = torch.nn.functional.mse_loss(pred_sc, train_sc)
            
            # 3. Normalize the sensitivity loss by the MLL's magnitude
            # This is the key step to prevent gradient domination.
            # We use .detach() so the scaling doesn't affect the MLL's own gradient.
            mll_val = mll.detach().abs()
            
            # We add a small epsilon to prevent division by zero
            scaled_loss = self.lambda_reg * loss_sc / (mll_val + 1e-8)
            # print(scaled_loss.item())
            
            # 4. Return the balanced, regularized MLL
            return mll - scaled_loss
        else:
            return mll
