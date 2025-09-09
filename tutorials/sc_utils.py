import torch
from j_computer import batchJacobian_AD
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# def train_with_sc(
#     model,
#     mll,
#     train_X,
#     train_Y,
#     train_SC,
#     lambda_reg=1.0,
#     iterations=200,
#     learning_rate=0.1,
# ):
#     """
#     Custom training loop to train a ModelListGP set of GP models with SC
#     regularization.

#     Args:
#         model: The ModelListGP to be trained.
#         mll: The MarginalLogLikelihood loss for the model.
#         train_X: The training inputs.
#         train_Y: The training targets (function values).
#         train_SC: The target sensitivities (Jacobians).
#         lambda_reg: The weight of the Jacobian regularization term.
#         iterations: The number of optimization steps.
#         learning_rate: The learning rate for the Adam optimizer.
#     """
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#     print(f"Training with Jacobian Regularization = {lambda_reg})...")

#     for i in range(iterations):
#         optimizer.zero_grad()

#         output = model(*model.train_inputs)

#         mll_loss = -mll(output, model.train_targets).sum()

#         train_X_grad = train_X.clone().requires_grad_(True)
        
#         # Get the posterior mean from the model at the original training points
#         posterior = model.posterior(train_X_grad)
#         model_sc = batchJacobian_AD(posterior.mean, train_X_grad, True, True)

#         jacobian_loss = torch.nn.functional.mse_loss(model_sc, train_SC)

#         total_loss = mll_loss + lambda_reg * jacobian_loss

#         total_loss.backward()
#         optimizer.step()

#         if (i + 1) % 20 == 0:
#             print(
#                 f"Iter {i+1}/{iterations} - "
#                 f"Total Loss: {total_loss.item():.3f} | "
#                 f"MLL Loss: {mll_loss.item():.3f} | "
#                 f"Jacobian Loss: {jacobian_loss.item():.3f}"
#             )
    
#     # Put the model back in evaluation mode
#     model.eval()
#     print("Training complete.")
#     return model



class SCMarginalLogLikelihood(SumMarginalLogLikelihood):
    """A custom MLL class that adds a SC regularization term to the loss.
    
    This class should be compatible with botorch.fit.fit_gpytorch_mll.
    """
    def __init__(self, likelihood, model, train_X, train_SC, lambda_reg=1.0):
        """
        Args:
            likelihood: The model's likelihood.
            model: The model to be trained (a ModelListGP).
            train_X: The original training inputs (un-augmented).
            train_SC: The target sensitivities (Jacobians).
            lambda_reg: The weight of the Jacobian regularization term.
        """
        super().__init__(likelihood, model)
        self.train_X = train_X
        self.train_SC = train_SC
        self.lambda_reg = lambda_reg

    def forward(self, model_outputs, targets):
        """Calculates the total loss = -MLL + lambda * Jacobian_Loss.
        
        `fit_gpytorch_mll` will call this method repeatedly during optimization.
        """
        # Calculate normal MLL loss from parent class (Maximize MLL, so minimize its negative)
        mll_loss = super().forward(model_outputs, targets)

        # Calc SC
        self.model.train()
        train_X_grad = self.train_X.clone().requires_grad_(True)
        
        posterior = self.model.posterior(train_X_grad)  # posterior mean
        
        pred_sc = batchJacobian_AD(posterior, train_X_grad, True, True)

        sc_loss = torch.nn.functional.mse_loss(
            pred_sc.squeeze(-2), self.train_SC
        )

        # fit_gpytorch_mll will minimize this
        return - mll_loss + self.lambda_reg * sc_loss
