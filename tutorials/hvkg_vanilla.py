import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_objective_multi_fidelity import MOMFBraninCurrin

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Very small noise values detected")



verbose = True  # set to True to see output during optimization

SMOKE_TEST = True #os.environ.get("SMOKE_TEST")




tkwargs = {  # Tkwargs is a dictionary contaning data about data type and data device
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

if SMOKE_TEST:
    print("Running in smoke test mode.")

BC = MOMFBraninCurrin(negate=True).to(**tkwargs)
dim_x = BC.dim
dim_y = BC.num_objectives

ref_point = torch.zeros(dim_y, **tkwargs) 
print(f"Reference point used: {ref_point}")  # Should be (0,0) for this problem


BATCH_SIZE = 1  # For batch optimization, BATCH_SIZE should be greater than 1
# This evaluation budget is set to be very low to make the notebook run fast. This should be much higher.
EVAL_BUDGET = 2  # in terms of the number of full-fidelity evaluations. ##### Tried 5 and didn't work
n_INIT = 2  # Initialization budget in terms of the number of full-fidelity evaluations
# Number of Monte Carlo samples, used to approximate MOMF
MC_SAMPLES = 2 if SMOKE_TEST else 128
# Number of restart points for multi-start optimization
NUM_RESTARTS = 2 if SMOKE_TEST else 10
# Number of raw samples for initial point selection heuristic
RAW_SAMPLES = 4 if SMOKE_TEST else 512

standard_bounds = torch.zeros(2, dim_x, **tkwargs)
standard_bounds[1] = 1
# mapping from index to target fidelity (highest fidelity)
target_fidelities = {2: 1.0}


from math import exp


def cost_func(x):
    """A simple exponential cost function."""
    exp_arg = torch.tensor(4.8, **tkwargs)
    val = torch.exp(exp_arg * x)
    return val


# Displaying the min and max (fidelity) costs for this optimization
print(f"Min Cost: {cost_func(0)}")  # Measuring fidelity from s = 0 to 1
print(f"Max Cost: {cost_func(1)}")


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


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior


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
    train_x = train_x[:-1]

    train_obj = BC(train_x)  ## Get synthetic data

    return train_x, train_obj


def initialize_model(train_x, train_obj, state_dict=None):
    """Initializes a ModelList with Matern 5/2 Kernel and returns the model and its MLL.

    Note: a batched model could also be used here.
    """
    models = []
    for i in range(train_obj.shape[-1]):
        m = SingleTaskGP(
            train_x,
            train_obj[:, i : i + 1],
            train_Yvar=torch.full_like(train_obj[:, i : i + 1], 1e-6),
            covar_module=ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_x.shape[-1],
                    lengthscale_prior=GammaPrior(2.0, 2.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            ),
        )
        models.append(m)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict=state_dict)
    return mll, model


from botorch.acquisition.multi_objective.multi_fidelity import MOMF
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import unnormalize

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.deterministic import GenericDeterministicModel
from torch import Tensor

NUM_INNER_MC_SAMPLES = 2 if SMOKE_TEST else 32
NUM_PARETO = 1 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 32  #8  ##### changed to 32
 

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


normalized_target_fidelities = {}
for idx, fidelity in target_fidelities.items():
    lb = standard_bounds[0, idx].item()
    ub = standard_bounds[1, idx].item()
    normalized_target_fidelities[idx] = (fidelity - lb) / (ub - lb)
project_d = dim_x


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
    new_obj = BC(new_x)
    return new_x, new_obj


from botorch import fit_gpytorch_mll



torch.manual_seed(0)
train_x_kg, train_obj_kg = gen_init_data(n_INIT)
MF_n_INIT = train_x_kg.shape[0]
iteration = 0
total_cost = cost_callable(train_x_kg).sum().item()

mse_over_iter = []

while total_cost < EVAL_BUDGET * cost_func(1):
    if verbose:
        print(f"cost: {total_cost}")

        with open("/storage/home/lgl5139/work/code/iclr25/hvkg_dev/out/hvkg_bc_cost.txt", "a") as f:
            f.write(f"Iteration {iteration}, Cost: {total_cost}\n")

    # reinitialize the models so they are ready for fitting on next iteration
    mll, model = initialize_model(normalize(train_x_kg, BC.bounds), train_obj_kg)

    fit_gpytorch_mll(mll=mll)  # Fit the model

    # model.eval()
    # with torch.no_grad():
    #     preds = model(*model.train_inputs)

    # preds_list = []
    # for i, mvn in enumerate(preds):
    #     mean = mvn.mean      # tensor of shape (48,)
    #     # print(f"Objective {i} mean:", mean)  # print first 5
    #     preds_list.append(mean.detach().cpu())

    new_x, new_obj = optimize_HVKG_and_get_obs(
        model=model,
        ref_point=ref_point,
        standard_bounds=standard_bounds,
        BATCH_SIZE=BATCH_SIZE,
        cost_call=cost_callable,
    )
    
    # mse = torch.mean((preds - train_obj_kg) ** 2).item()
    # mse_over_iter.append(mse)

    # Updating train_x and train_obj
    train_x_kg = torch.cat([train_x_kg, new_x], dim=0)
    train_obj_kg = torch.cat([train_obj_kg, new_obj], dim=0)
    iteration += 1
    total_cost += cost_callable(new_x).sum().item()

if total_cost >= EVAL_BUDGET * cost_func(1):
    print(f"Cost threshold exceeded: {total_cost} >> {EVAL_BUDGET * cost_func(1)}")

 
# dif = np.abs(model.train_targets[0].cpu() - preds_list[0])
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(model.train_targets[0].cpu(), preds_list[0], "o", color="blue")
# # plt.plot(dif, range(len(dif)), "x", color="red")


# # add 1:1 line
# min_val = min(model.train_targets[0].min().item(), preds_list[0].min().item())
# max_val = max(model.train_targets[0].max().item(), preds_list[0].max().item())
# plt.plot([min_val, max_val], [min_val, max_val], "--", color="black")

# plt.ylabel("Surrogate")
# plt.xlabel("Target")
# plt.title("Surrogate vs Target Objective")
# plt.savefig("/projects/mhpi/leoglonz/project_silmaril/surrogate_examples/hvkg/results/surrogate_temp.png")



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




hvs_kg = []
costs = []
for i in range(MF_n_INIT, train_x_kg.shape[0] + 1, 5):

    mll, model = initialize_model(
        normalize(train_x_kg[:i], BC.bounds), train_obj_kg[:i]
    )
    fit_gpytorch_mll(mll)
    hypervolume = get_pareto(model, project=project, non_fidelity_indices=[0, 1])
    hvs_kg.append(hypervolume)
    costs.append(cost_callable(train_x_kg[:i]).sum().item())



plt.plot(
    costs, np.log10(BC.max_hv - np.array(hvs_kg)), "--", marker="d", ms=10, label="HVKG"
)
plt.ylabel("Log Inference Hypervolume Regret")
plt.xlabel("Cost")
plt.legend()

plt.savefig("/storage/home/lgl5139/work/code/iclr25/hvkg_dev/out/figs/hvkg.png")
