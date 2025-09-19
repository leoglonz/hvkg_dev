import os
import warnings
from typing import Callable, Dict

import numpy as np
import torch
from botorch.test_functions.multi_objective_multi_fidelity import MOMFBraninCurrin
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import qMultiFidelityHypervolumeKnowledgeGradient
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.deterministic import GenericDeterministicModel
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import _get_hv_value_function
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch import settings
from math import exp
from torch import Tensor
from botorch import fit_gpytorch_mll
import matplotlib.pyplot as plt


# custom imports
from j_computer import batchJacobian_AD
from sc_mll import SensitivityAwareGP, SensitivityAwareMLL

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Very small noise values detected")

# =============================================================================
# Experiment Constants
# =============================================================================
SMOKE_TEST = False
NUM_REPLICATIONS = 20 # Total number of random seeds to run
SEEDS = [0, 42, 259068, 355549, 369813, 467073, 488127, 561897, 593786, 831920, 905678, 998244, 222779, 41371, 649186, 287910, 510955, 490956, 955276, 319515] #, 806017]

# SC ----
USE_SC = True

# --- BO Loop Settings ---
BATCH_SIZE = 1
EVAL_BUDGET = 2  # In terms of the number of full-fidelity evaluations
N_INIT = 2       # Initialization budget

# --- Acqf Settings ---
MC_SAMPLES = 2 if SMOKE_TEST else 128
NUM_RESTARTS = 2 if SMOKE_TEST else 10
RAW_SAMPLES = 4 if SMOKE_TEST else 512
NUM_INNER_MC_SAMPLES = 32
NUM_PARETO = 1 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 32  # originally 8

# --- Device and dtype Settings ---
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

EVAL_RESOLUTION = 5

# =============================================================================
# Problem Setup
# =============================================================================
BC = MOMFBraninCurrin(negate=True).to(**tkwargs)
dim_x = BC.dim
dim_y = BC.num_objectives
ref_point = torch.zeros(dim_y, **tkwargs) 

standard_bounds = torch.zeros(2, dim_x, **tkwargs)
standard_bounds[1] = 1
target_fidelities = {2: 1.0} # mapping from index to target fidelity


normalized_target_fidelities = {}
for idx, fidelity in target_fidelities.items():
    lb = standard_bounds[0, idx].item()
    ub = standard_bounds[1, idx].item()
    normalized_target_fidelities[idx] = (fidelity - lb) / (ub - lb)
project_d = dim_x


# =============================================================================
# Helper Functions
# =============================================================================
def cost_func(x):
    """A simple exponential cost function."""
    exp_arg = torch.tensor(4.8, **tkwargs)
    val = torch.exp(exp_arg * x)
    return val


def cost_callable(X: torch.Tensor) -> torch.Tensor:
    r"""Wrapper for the cost function that takes care of shaping
    input and output arrays for interfacing with cost_func.
    This is passed as a callable function to MOMF.

    Args:
        X: A `batch_shape x q x d`-dim Tensor
    Returns:
        Cost `batch_shape x q x m`-dim Tensor of cost generated
        from fidelity dimension using cost_func.
    """

    return cost_func(X[..., -1:])


def inv_transform(u):
    # define inverse transform to sample from the probability distribution with
    # PDF proportional to 1/(c(x))
    # u is a uniform(0,1) rv
    return 5 / 24 * torch.log(-exp(24 / 5) / (exp(24 / 5) * u - u - exp(24 / 5)))


def gen_init_data(n: int):
    r"""
    Generates the initial data. Sample fidelities inversely proportional to cost.
    """
    # total cost budget is n
    train_x = torch.empty(
        0, BC.bounds.shape[1], dtype=BC.bounds.dtype, device=BC.bounds.device
    )
    total_cost = 0
    # assume target fidelity is 1
    total_cost_limit = (
        n
        * cost_callable(
            torch.ones(
                1, BC.bounds.shape[1], dtype=BC.bounds.dtype, device=BC.bounds.device
            )
        ).item()
    )
    while total_cost < total_cost_limit:
        new_x = torch.rand(
            1, BC.bounds.shape[1], dtype=BC.bounds.dtype, device=BC.bounds.device
        )
        new_x[:, -1] = inv_transform(new_x[:, -1])
        total_cost += cost_callable(new_x)
        train_x = torch.cat([train_x, new_x], dim=0)

    ############################ For SC
    train_x = train_x[:-1]  # [x1, x2, s]
    norm_x = normalize(train_x, BC.bounds)

    train_s = norm_x[:, -1]
    train_s = train_s.requires_grad_(True)
    norm_x = torch.cat([norm_x[:, :-1], train_s.unsqueeze(-1)], dim=-1)

    train_obj = BC(train_x)  ## Get synthetic data
    norm_obj = BC(norm_x)
    train_sc = batchJacobian_AD(norm_obj, train_s, True, True)  # SC on s- fidelity

    train_x = train_x.detach()
    train_obj = train_obj.detach()
    train_sc = train_sc.detach()
    ############################

    return train_x, train_obj, train_sc


def initialize_model(train_x, train_obj, train_sc, state_dict=None, eval=False):
    """
    Initializes a ModelList with standardized outputs for stability.
    """
    models = []
    for i in range(train_obj.shape[-1]):
        subset_frac = 0.5
        if i == 0:
            lambda_reg = 1e-1
        else:
            lambda_reg = 1e1
        
        if (not USE_SC) or (eval):
            lambda_reg = 0

        train_Y = train_obj[:, i : i + 1]
        train_S = train_sc[:, i : i + 1]

        # 1. Create and fit an outcome transform to standardize the objectives
        outcome_transform = Standardize(m=1) # m=1 for single-output GP
        # The transform is fitted within the model constructor, so we just pass it
        
        # 2. We need to scale our sensitivity targets to match the standardized space.
        # We can approximate the std. dev. directly from the training data.
        Y_std = train_Y.std()
        if Y_std < 1e-6: # Avoid division by zero
            Y_std = 1.0
        
        scaled_train_S = train_S / Y_std

        m = SensitivityAwareGP(
            train_x,
            train_Y, # The model will handle standardization internally
            scaled_train_S, # Pass the correctly scaled sensitivity targets
            # train_Yvar=torch.full_like(train_obj[:, i : i + 1], 1e-6),
            lambda_reg=lambda_reg,
            subset_frac=subset_frac,
            outcome_transform=outcome_transform, # Pass the transform to the model
            covar_module=ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_x.shape[-1],
                    lengthscale_prior=GammaPrior(2.0, 2.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            ),
        )
        m.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        models.append(m)
        
    model = ModelListGP(*models)
    
    mll = SumMarginalLogLikelihood(
        model.likelihood, 
        model, 
        mll_cls=SensitivityAwareMLL,
        # With standardized outputs, lambda=1.0 is a good starting point
    )
    
    if state_dict is not None:
        model.load_state_dict(state_dict=state_dict)
    return mll, model


def get_current_value(
    model: SingleTaskGP,
    ref_point: torch.Tensor,
    bounds: torch.Tensor,
    normalized_target_fidelities: Dict[int, float],
):
    """Helper to get the hypervolume of the current hypervolume
    maximizing set.
    """
    fidelity_dims, fidelity_targets = zip(*normalized_target_fidelities.items())
    # optimize
    non_fidelity_dims = list(set(range(dim_x)) - set(fidelity_dims))
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=_get_hv_value_function(
            model=model,
            ref_point=ref_point,
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([NUM_INNER_MC_SAMPLES]),
            ),
            use_posterior_mean=True,
        ),
        d=dim_x,
        columns=fidelity_dims,
        values=fidelity_targets,
    )
    # optimize
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, non_fidelity_dims],
        q=NUM_PARETO,
        num_restarts=1,
        raw_samples=2 * RAW_SAMPLES,
        return_best_only=True,
        options={
            "nonnegative": True,
            "maxiter": 3 if SMOKE_TEST else 200,
        },
    )
    return current_value


def project(X: Tensor) -> Tensor:
    return project_to_target_fidelity(
        X=X,
        d=project_d,
        target_fidelities=normalized_target_fidelities,
    )


def optimize_HVKG_and_get_obs(
    model: SingleTaskGP,
    ref_point: torch.Tensor,
    standard_bounds: torch.Tensor,
    BATCH_SIZE: int,
    cost_call: Callable[[torch.Tensor], torch.Tensor],
    normalized_target_fidelities,
):
    """Utility to initialize and optimize HVKG."""
    cost_model = GenericDeterministicModel(cost_call)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    current_value = get_current_value(
        model=model,
        ref_point=ref_point,
        bounds=standard_bounds,
        normalized_target_fidelities=normalized_target_fidelities,
    )

    acq_func = qMultiFidelityHypervolumeKnowledgeGradient(
        model=model,
        ref_point=ref_point,  # use known reference point
        num_fantasies=NUM_FANTASIES,
        num_pareto=NUM_PARETO,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        target_fidelities=normalized_target_fidelities,
        project=project,
    )
    # Optimization
    candidates, vals = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=1,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={
            "batch_limit": 5,
            "maxiter": 3 if SMOKE_TEST else 200,
        },
    )
    # if the AF val is 0, set the fidelity parameter to zero
    if vals.item() == 0.0:
        candidates[:, -1] = 0.0
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=BC.bounds)
    new_norm_x = candidates.detach()

    new_s = new_norm_x[:, -1]
    new_s = new_s.requires_grad_(True)
    new_norm_x = torch.cat([new_norm_x[:, :-1], new_s.unsqueeze(-1)], dim=-1)

    new_obj = BC(new_x)
    new_norm_obj = BC(new_norm_x)
    new_sc = batchJacobian_AD(new_norm_obj, new_s, True, True)

    new_x = new_x.detach()
    new_obj = new_obj.detach()
    new_sc = new_sc.detach()

    return new_x, new_obj, new_sc


def train_model_adam(mll, training_steps=400, learning_rate=0.01):
    """
    Custom two-phase training loop for improved stability and performance.
    """
    # Phase 1: Pre-training on MLL only
    # This finds good general hyperparameters before introducing the sensitive term.
    pretrain_steps = int(training_steps * 0.75)
    
    # Phase 2: Fine-tuning with the full, regularized loss
    finetune_steps = training_steps - pretrain_steps

    optimizer = torch.optim.Adam(mll.model.parameters(), lr=learning_rate)
    model = mll.model
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    model.train()
    
    # --- Phase 1: MLL Pre-training ---
    print("--- Starting Phase 1: MLL Pre-training ---")
    for i in range(pretrain_steps):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        
        # We need to get the MLL from each individual model in the ModelListGP
        # and then sum them up. We can't call mll() directly here.
        pretrain_loss = 0
        for j, single_model in enumerate(model.models):
            pretrain_loss -= single_model.likelihood(output[j], single_model.train_targets).mean.sum()

        if not torch.isfinite(pretrain_loss):
            print(f"Warning: Non-finite loss in Phase 1 at step {i}. Stopping.")
            break
        pretrain_loss.backward()
        optimizer.step()
        scheduler.step()

    # --- Phase 2: Sensitivity Fine-tuning ---
    print("--- Starting Phase 2: Sensitivity Fine-tuning ---")
    for i in range(finetune_steps):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        # Now use the full, regularized MLL object
        loss = -mll(output, model.train_targets)
        
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss in Phase 2 at step {i}. Stopping.")
            break
        loss.backward()
        optimizer.step()
        scheduler.step()



from botorch.utils.multi_objective.pareto import (
    _is_non_dominated_loop,
    is_non_dominated,
)
from gpytorch import settings

try:
    # Note: These are the pymoo 0.6+ imports, if you happen to be stuck on
    # an older pymoo version you need to replace them with the ones below.
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.termination.max_gen import MaximumGenerationTermination

    # from pymoo.algorithms.nsga2 import NSGA2
    # from pymoo.model.problem import Problem
    # from pymoo.util.termination.max_gen import MaximumGenerationTermination

    def get_pareto(
        model,
        non_fidelity_indices,
        project,
        population_size=20 if SMOKE_TEST else 250,
        max_gen=10 if SMOKE_TEST else 100,
        is_mf_model=True,
    ):
        """Optimize the posterior mean using NSGA-II."""
        tkwargs = {
            "dtype": BC.ref_point.dtype,
            "device": BC.ref_point.device,
        }
        dim = len(non_fidelity_indices)

        class PosteriorMeanPymooProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=dim,
                    n_obj=BC.num_objectives,
                    type_var=np.double,
                )
                self.xl = np.zeros(dim)
                self.xu = np.ones(dim)

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**tkwargs)
                if is_mf_model:
                    X = project(X)
                with torch.no_grad():
                    with settings.cholesky_max_tries(9):
                        # eval in batch mode
                        y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                out["F"] = -y.cpu().numpy()

        pymoo_problem = PosteriorMeanPymooProblem()
        algorithm = NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True,
        )
        res = minimize(
            pymoo_problem,
            algorithm,
            termination=MaximumGenerationTermination(max_gen),
            seed=0,  # fix seed
            verbose=False,
        )
        X = torch.tensor(
            res.X,
            **tkwargs,
        )
        # project to full fidelity
        if is_mf_model:
            if project is not None:
                X = project(X)
        # determine Pareto set of designs under model
        with torch.no_grad():
            preds = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
        pareto_mask = is_non_dominated(preds)
        X = X[pareto_mask]
        # evaluate Pareto set of designs on true function and compute hypervolume
        if not is_mf_model:
            X = project(X)
        X = unnormalize(X, BC.bounds)
        Y = BC(X)
        # compute HV
        partitioning = FastNondominatedPartitioning(ref_point=BC.ref_point, Y=Y)
        return partitioning.compute_hypervolume().item()

except ImportError:
    NUM_DISCRETE_POINTS = 10 if SMOKE_TEST else 100000
    CHUNK_SIZE = 512

    def get_pareto(
        model,
        non_fidelity_indices,
        project,
        population_size=20 if SMOKE_TEST else 250,
        max_gen=10 if SMOKE_TEST else 100,
        is_mf_model=True,
    ):
        """Optimize the posterior mean over a discrete set."""
        tkwargs = {
            "dtype": BC.ref_point.dtype,
            "device": BC.ref_point.device,
        }
        dim_x = BC.dim

        discrete_set = torch.rand(NUM_DISCRETE_POINTS, dim_x - 1, **tkwargs)
        if is_mf_model:
            discrete_set = project(discrete_set)
        discrete_set[:, -1] = 1.0  # set to target fidelity
        with torch.no_grad():
            preds_list = []
            for start in range(0, NUM_DISCRETE_POINTS, CHUNK_SIZE):
                preds = model.posterior(
                    discrete_set[start : start + CHUNK_SIZE].unsqueeze(-2)
                ).mean.squeeze(-2)
                preds_list.append(preds)
            preds = torch.cat(preds_list, dim=0)
            pareto_mask = _is_non_dominated_loop(preds)
            pareto_X = discrete_set[pareto_mask]
        if not is_mf_model:
            pareto_X = project(pareto_X)
        pareto_X = unnormalize(pareto_X, BC.bounds)
        Y = BC(pareto_X)
        # compute HV
        partitioning = FastNondominatedPartitioning(ref_point=BC.ref_point, Y=Y)
        return partitioning.compute_hypervolume().item()



# =============================================================================
# Main Experiment Function
# =============================================================================

def run_full_experiment(seed: int, output_dir: str = "out"):
    print(f"\n--- Starting Experiment for Seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Create a unique directory for this run's results ---
    if USE_SC:
        run_output_dir = os.path.join(output_dir, 'with_sc')
    else:
        run_output_dir = os.path.join(output_dir, 'vanilla')
    
    run_output_dir = os.path.join(run_output_dir, f"run_{seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    training_costs_path = os.path.join(run_output_dir, "training_costs.txt")
    eval_results_path = os.path.join(run_output_dir, "evaluation_results.txt")


    # --- Main script ---
    train_x_kg, train_obj_kg, train_sc_kg = gen_init_data(N_INIT)
    MF_n_INIT = train_x_kg.shape[0]
    total_cost = cost_callable(train_x_kg).sum().item()

    # Store the cumulative cost at each iter
    training_costs_over_time = [total_cost]

    iteration = 0

    # --- Main BO Loop ---
    while total_cost < EVAL_BUDGET * cost_func(1):
        print(f"Seed {seed}, Iteration {iteration}, Cost: {total_cost:.2f}")

        # reinitialize the models so they are ready for fitting on next iteration
        mll, model = initialize_model(normalize(train_x_kg, BC.bounds), train_obj_kg, train_sc_kg)

        if USE_SC:
            train_model_adam(mll=mll)
        else:
            fit_gpytorch_mll(mll=mll)  # Fit the model

        new_x, new_obj, new_sc = optimize_HVKG_and_get_obs(
            model=model,
            ref_point=ref_point,
            standard_bounds=standard_bounds,
            BATCH_SIZE=BATCH_SIZE,
            cost_call=cost_callable,
            normalized_target_fidelities=normalized_target_fidelities,
        )

        # Updating train data
        train_x_kg = torch.cat([train_x_kg, new_x], dim=0)
        train_obj_kg = torch.cat([train_obj_kg, new_obj], dim=0)
        train_sc_kg = torch.cat([train_sc_kg, new_sc], dim=0)
        
        # Update cost and iteration count
        iteration += 1
        total_cost += cost_callable(new_x).sum().item()
        training_costs_over_time.append(total_cost)

    print(f"Seed {seed} finished. Final cost: {total_cost:.2f}")
    print(f"Cost threshold exceeded: {total_cost} >> {EVAL_BUDGET * cost_func(1)}")

    # --- Save training costs ---
    np.savetxt(training_costs_path, np.array(training_costs_over_time), fmt="%.6f")


    # --- Evaluation Phase ---
    print(f"--- Seed {seed}: Starting Evaluation Phase ---")
    hvs_kg = []
    costs_at_eval = []

    # Checkpoint at every 5 iterations
    for i in range(MF_n_INIT, train_x_kg.shape[0] + 1, EVAL_RESOLUTION):
        mll, model = initialize_model(
            normalize(train_x_kg[:i], BC.bounds), train_obj_kg[:i], train_sc_kg[:i], eval=True,
        )

        # if USE_SC:
        #     train_model_adam(mll=mll)
        # else:
        fit_gpytorch_mll(mll=mll)  # Fit the model

        hypervolume = get_pareto(model, project=project, non_fidelity_indices=[0, 1])
        hvs_kg.append(hypervolume)
        costs_at_eval.append(cost_callable(train_x_kg[:i]).sum().item())

    costs_at_eval = np.array(costs_at_eval)
    regret = np.log10(BC.max_hv - np.array(hvs_kg))

    # --- Save evaluation results ---
    eval_results = np.stack([costs_at_eval, regret], axis=1)
    np.savetxt(eval_results_path, eval_results, fmt="%.6f", header="Cost Regret")

    plt.plot(
        costs_at_eval, np.log10(BC.max_hv - np.array(hvs_kg)), "--", marker="d", ms=10, label="HVKG"
    )
    plt.ylabel("Log Inference Hypervolume Regret")
    plt.xlabel("Cost")
    plt.savefig(os.path.join(run_output_dir, 'cost-regret.png'))
    plt.close()
    print(f"Results for seed {seed} saved to {run_output_dir}")



# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if SMOKE_TEST:
        print("Running in smoke test mode (1 replication).")
        run_full_experiment(seed=0)
    else:
        print(f"Starting full experiment with {NUM_REPLICATIONS} replications.")
        for i in range(NUM_REPLICATIONS):
            run_full_experiment(seed=SEEDS[i])
    print("\n--- All experiments complete. ---")
