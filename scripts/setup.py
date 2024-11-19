# Copyright (C) 2024 Richard Stiskalek, Nicholas Choustikov
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from os.path import isdir, join

import photonion
from joblib import load
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


###############################################################################
#                       Baked in hyperparameters                             #
###############################################################################

fesc_min = 1e-8         # Minimum fesc value to be considered
test_fraction = 0.2     # Fraction of the data to be used for testing
ntrials_sklearn = 5000  # Number of trials for sklearn hyperparameter optim
nfolds_sklearn = 5      # Number of folds for cross-validation in sklearn
ntrials_sbi = 1000      # Number of trials for SBI hyperparameter optim
nfolds_sbi = 10         # Number of folds for cross-validation in SBI
perm_repeats = 10       # Number of repeats for permutation importance

###############################################################################
#       Default hyperparameters and their grids for scikit-learn              #
###############################################################################


def regressor_default(model_str):
    """
    Get a regressor with default hyperparameters.

    Parameters
    ----------
    model_str : str
        The regressor name.

    Returns
    -------
    dict
    """
    if model_str == "ET":
        return ExtraTreesRegressor(bootstrap=True, n_jobs=1)
    elif model_str == "linear":
        return LinearRegression(n_jobs=1)
    else:
        raise ValueError(f"Invalid model `{model_str}`.")


###############################################################################
#                           Loading best models                               #
###############################################################################


def load_best_model(basepath, model_str, target, kind):
   """
   Load the best model for a given regressor. Note that the model is not
   fitted.

   Parameters
   ----------
   basepath : str
      Path to saved models.
   model_str : str
      Regressor name.
   target : str
      Target variable.
   kind : str
      Data set kind.

   Returns
   -------
   sklearn.pipeline.Pipeline
   """
   if not isdir(basepath):
      raise ValueError(f"Invalid base path `{basepath}`.")
   
   reg = regressor_default(model_str)
   include_logit = True if target == "fesc" else False
   pipe = photonion.make_pipeline(reg, include_logit)

   # Linear model has no hyperparameters
   if model_str == "linear":
      return pipe

   fname_hyper = join(basepath, "results",
                     f"hyper_{target}_{kind}_{model_str}.pkl")
   pipe.set_params(**load(fname_hyper)["best_params"])

   return pipe


def sbi_read_best_params(basepath, target, kind):
   """
   Load the best hyperparameters for SBI from `run_hyper_sbi.py`.

   Parameters
   ----------
   basepath : str
      Path to saved models.
   target : str
      Target variable.
   kind : str
      Data set kind.

   Returns
   -------
   train_args : dict
      Training hyperparameters.
   model_args : dict
      Model hyperparameters.
   """
   if not isdir(basepath):
      raise ValueError(f"Invalid base path `{basepath}`.")

   fname_hyper = join(basepath, "results", f"hyper_{target}_{kind}_SBI.pkl")
   params = load(fname_hyper)["best_params"]

   train_params = ["training_batch_size", "learning_rate",
                  "validation_fraction", "stop_after_epochs",
                  "clip_max_norm"]
   train_args = {}
   for param in train_params:
      if param in params:
         train_args[param] = params[param]

   model_params = ["hidden_features", "num_components"]
   model_args = {}
   for param in model_params:
      if param in params:
         model_args[param] = params[param]

   return train_args, model_args
