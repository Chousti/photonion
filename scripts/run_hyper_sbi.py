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
"""
Script for running hyperparameter optimization for SBI LTU-ILI pipeline. The
Optuna hyperparameter optimization library is parallelized but MPI is not
supported. The hyperparameter optimization is performed across the entire
data set.
"""
import warnings
from argparse import ArgumentParser
from os.path import join

import joblib
import numpy as np
import photonion
from optuna import create_study
from sklearn.model_selection import GroupKFold

import setup

from run_hyper_sklearn import t

from joblib import delayed, Parallel


def evaluate_fold(train_index, test_index, X, y, target_scaling, train_args,
                  model_args):
    """
    Evaluate a single fold of the cross-validation for fixed hyperparameters.
    """
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]

    # Initialize and fit your model here as before
    reg = photonion.SBIRegressor(
        target_scaling, num_nets=3, train_args=train_args,
        model_args=model_args)
    reg.fit(Xtrain, ytrain, verbose=False)
    score = np.mean(reg.log_prob(Xtest, ytest, False))

    # Continue with the PIT calculation and adjust the score
    pit = reg.calculate_pit(Xtest, ytest, n_samples=5000, verbose=False)
    dpit_max = np.max(np.abs(pit - np.linspace(0, 1, len(pit))))
    score += - 0.5 * np.log(dpit_max)

    return score


def objective(trial, X, y, groups, target_scaling):
    """Optuna objective function for SBI hyperparameter optimization."""
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    hidden_features = trial.suggest_int("hidden_features", 12, 200)
    num_components = trial.suggest_int("num_components", 2, 16)
    training_batch_size = trial.suggest_int("training_batch_size", 32, 128)
    stop_after_epochs = trial.suggest_int("stop_after_epochs", 10, 30)

    train_args = {"learning_rate": learning_rate,
                  "stop_after_epochs": stop_after_epochs,
                  "training_batch_size": training_batch_size}
    model_args = {"hidden_features": hidden_features,
                  "num_components": num_components}

    cv = GroupKFold(n_splits=setup.nfolds_sbi)
    scores = Parallel(n_jobs=10)(delayed(evaluate_fold)(train_index, test_index, X, y, target_scaling, train_args, model_args)  # noqa
                                 for train_index, test_index in cv.split(X, y, groups))                                         # noqa
    score = np.mean(scores)

    print(f"\n{t()}: finished trial with mean score {score:.5f}.", flush=True)
    print(flush=True)
    return score


if __name__ == "__main__":
    parser = ArgumentParser(description="Run SBI hyperparameter optimization.")
    parser.add_argument("--target", type=str, required=True,
                        choices=["fesc", "nion"],
                        help="The target variable to predict.")
    parser.add_argument("--kind", type=str, required=True,
                        choices=["ALL", "JADES"], help="Dataset to use.")
    parser.add_argument("--nthreads", type=int, default=1,
                        help="Number of threads to use.")
    parser.add_argument("--machine", type=str, required=True,
                        help="The machine name to get paths.")
    args = parser.parse_args()

    basepath = setup.get_basepath(args.machine)

    # Load in the X and y data. Train-test split is done within the objective
    print(f"{t()}: loading data...", flush=True)
    frame = photonion.SPHINXData(join(basepath, "data", "all_basic_data.csv"))
    Xtrain, Xtest, ytrain, ytest, train_groups, test_groups, __ = frame.make_Xy(  # noqa
        setup.test_fraction, args.target, args.kind, setup.fesc_min, 0)
    X = np.concatenate((Xtrain, Xtest), axis=0)
    y = np.concatenate((ytrain, ytest), axis=0)
    groups = np.concatenate((train_groups, test_groups), axis=0)

    target_scaling = "logit_standard" if args.target == "fesc" else "standard"

    # Create the study and run the hyperparameter optimization.
    study = create_study(direction="maximize")

    def objective_wargs(trial):
        return objective(trial, X, y, groups, target_scaling)

    print(f"{t()}: Starting hyperparameter optimization.", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        study.optimize(objective_wargs, n_trials=setup.ntrials_sbi,
                       n_jobs=args.nthreads, show_progress_bar=True)

    print(f"{t()}: Hyperparameter optimization finished.", flush=True)
    print(f"Best value: {study.best_value:.5f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {str(k):<30} {v}")

    # Save the results.
    fout = join(basepath, "results",
                f"hyper_{args.target}_{args.kind}_SBI.pkl")
    print(f"{t()}: dumping results to `{fout}`.", flush=True)
    out = {"best_params": study.best_params, "study": study}
    joblib.dump(out, fout)

    print(f"{t()}: all finished!", flush=True)
