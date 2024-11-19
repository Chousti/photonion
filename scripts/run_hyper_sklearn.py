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
Script for running hyperparameter optimization for scikit-learn models. The
Optuna hyperparameter optimization library is parallelized but MPI is not
supported. The hyperparameter optimization is performed across the entire
data set, though the final model is trained on the training set only and
evaluated on the test set.
"""
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime
from os.path import join

import joblib
import numpy as np
import photonion
from optuna.distributions import (FloatDistribution, IntDistribution)
from optuna.integration import OptunaSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

import setup


def t():
    """
    Get the current time.

    Returns
    -------
    time
    """
    return datetime.now().strftime("%H:%M:%S")


def get_hyperparameter_grid(model_str, with_logit_transform):
    """
    Get the scikit-liearn pipeline hyperparameter grid for OptunaSearchCV.

    Parameters
    ----------
    model_str : str
        The regressor name.
    with_logit_transform : bool
        Whether the logit transform is included in the pipeline.

    Returns
    -------
    dict
    """
    if model_str == "ET":
        grid = {
            "n_estimators": IntDistribution(32, 512),
            "max_depth": IntDistribution(2, 48),
            "min_samples_split": IntDistribution(2, 128),
            "max_features": FloatDistribution(0.1, 0.99),
            "min_impurity_decrease": FloatDistribution(1e-16, 0.5, log=True),
            "ccp_alpha": FloatDistribution(1e-16, 0.5, log=True),
            "max_samples": FloatDistribution(0.1, 0.99),
            }
    else:
        raise ValueError(f"Invalid model `{model_str}`.")

    # Rename the keys to be prefixed with the regressor name
    renamed_grid = {}
    for key, value in grid.items():
        if with_logit_transform:
            renamed_grid[f"estimator__regressor__{key}"] = value
        else:
            renamed_grid[f"estimator__{key}"] = value

    return renamed_grid


if __name__ == "__main__":
    parser = ArgumentParser(description="Run hyperparameter optimization for scikit-learn models.")  # noqa
    parser.add_argument("--model", type=str, required=True, choices=["ET"],
                        help="The regressor model to optimize.")
    parser.add_argument("--target", type=str, required=True,
                        choices=["fesc", "nion"],
                        help="The target variable to predict.")
    parser.add_argument("--kind", type=str, required=True,
                        choices=["ALL", "JADES"], help="Dataset to use.")
    parser.add_argument("--loss", type=str, required=True,
                        choices=["KL", "MAE"],
                        help="The loss function to use.")
    parser.add_argument("--nthreads", type=int, default=1,
                        help="Number of threads to use.")
    parser.add_argument("--machine", type=str, required=True,
                        help="The machine name to get paths.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for the train-test split.")
    parser.add_argument("--optuna_seed", type=int, default=42,
                        help="Random seed for Optuna.")
    args = parser.parse_args()

    basepath = setup.get_basepath(args.machine)

    # Load in the X and y data
    print(f"{t()}: loading data...", flush=True)
    frame = photonion.SPHINXData(join(basepath, "data", "all_basic_data.csv"))
    Xtrain, Xtest, ytrain, ytest, indxs_train, indxs_test, feature_names = frame.make_Xy(  # noqa
        setup.test_fraction, args.target, args.kind, setup.fesc_min,
        args.split_seed)

    X = np.concatenate((Xtrain, Xtest), axis=0)
    y = np.concatenate((ytrain, ytest), axis=0)
    groups = np.concatenate((indxs_train, indxs_test), axis=0)

    include_logit = True if args.target == "fesc" else False
    reg = setup.regressor_default(args.model)
    pipe = photonion.make_pipeline(reg, include_logit)
    grid = get_hyperparameter_grid(args.model, include_logit)

    if args.loss == "KL" and args.target != "fesc":
        raise ValueError("KL loss is only valid for the `fesc` target.")

    def scoring(estimator, X_test, y_test):
        y_pred = estimator.predict(X_test)
        if args.loss == "KL":
            return -1 * photonion.KL_loss_fraction(y_test, y_pred)
        elif args.loss == "MAE" and args.target == "fesc":
            return -1 * photonion.MAE_log_log(y_test, y_pred)
        elif args.loss == "MAE" and args.target == "nion":
            return -1 * mean_absolute_error(y_test, y_pred)
        else:
            raise ValueError(f"Invalid loss function `{args.loss}`.")

    search = OptunaSearchCV(pipe, grid,
                            cv=GroupKFold(n_splits=setup.nfolds_sklearn),
                            scoring=scoring,
                            random_state=args.optuna_seed, n_jobs=-1,
                            n_trials=setup.ntrials_sklearn, verbose=0)

    print(f"{t()}: Starting hyperparameter optimization for `{args.model}`.",
          flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        search.fit(X, y, groups=groups)

    # Evaluate the best model.
    pipe.set_params(**search.best_params_)
    pipe.fit(Xtrain, ytrain)

    ytrain_pred, ytest_pred = pipe.predict(Xtrain), pipe.predict(Xtest)
    if args.target == "fesc":
        log_ytrain, log_ytest = np.log10(ytrain), np.log10(ytest)
        log_ytrain_pred = np.log10(ytrain_pred)
        log_ytest_pred = np.log10(ytest_pred)
    else:
        log_ytrain, log_ytest = ytrain, ytest
        log_ytrain_pred, log_ytest_pred = ytrain_pred, ytest_pred

    train_MAE = mean_absolute_error(log_ytrain, log_ytrain_pred)
    test_MAE = mean_absolute_error(log_ytest, log_ytest_pred)
    train_R2 = r2_score(log_ytrain, log_ytrain_pred)
    test_R2 = r2_score(log_ytest, log_ytest_pred)
    train_rho = spearmanr(ytrain, pipe.predict(Xtrain))[0]
    test_rho = spearmanr(ytest, pipe.predict(Xtest))[0]

    def summarize():
        print()
        print(f"--------------- Summary of best {args.model} model ---------------")  # noqa
        print(f"Train log-log MAE   {train_MAE:.5f}")
        print(f"Test log-log MAE    {test_MAE:.5f}")
        print(f"Train log-log R2    {train_R2:.5f}")
        print(f"Test log-log R2     {test_R2:.5f}")
        print(f"Train Spearman      {train_rho:.5f}")
        print(f"Test Spearman       {test_rho:.5f}")
        print()
        print("Split seed:", args.split_seed)
        print()

        print("Best hyperparameters:")
        for k, v in search.best_params_.items():
            print(f"    {str(k):<30} {v}")
        print()

        try:
            print("Feature importances:")
            if include_logit:
                fimp = pipe.named_steps["estimator"].regressor_.feature_importances_  # noqa
            else:
                fimp = pipe.named_steps["estimator"].feature_importances_
            for k, v in zip(feature_names, fimp):
                print(f"    {str(k):<20} {v:.5f}")
        except AttributeError:
            pass

        print(flush=True)

    summarize()

    # Save the results.
    fout = join(basepath, "results",
                f"hyper_{args.target}_{args.kind}_{args.model}.pkl")
    print(f"{t()}: dumping results to `{fout}`.", flush=True)
    out = {"trials": search.trials_, "best_params": search.best_params_}
    joblib.dump(out, fout)

    with open(fout.replace(".pkl", ".txt"), 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        summarize()
        sys.stdout = original_stdout

    print(f"{t()}: all finished!", flush=True)
