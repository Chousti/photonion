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
MPI script to fit the best model to many train-test splits and calculate the
performance metrics and feature importances.
"""
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import isdir, join
from time import time

import numpy as np
import photonion
from h5py import File
from mpi4py import MPI
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from taskmaster import work_delegation  # noqa
from tqdm import tqdm

import setup
from run_hyper_sklearn import t

###############################################################################
#                           Helper functions                                  #
###############################################################################


def mean_std(arr, axis=None):
    """
    Calculate the mean and standard deviation of an array.

    Parameters
    ----------
    arr : array-like
        The array.
    axis : int, optional
        The axis to calculate the mean and standard deviation along.

    Returns
    -------
    std, mean : float, float
    """
    return np.mean(arr, axis=axis), np.std(arr, axis=axis)


def get_split_seeds(seed, ntot, comm):
    """
    Get the random seeds for the train-test splits.

    Parameters
    ----------
    seed : int
        The initial random seed for the random number generator.
    nmax : int
        The total number of train-test splits.
    comm : MPI.Comm
        The MPI communicator.

    Returns
    -------
    split_seeds : 1-dimensional array
    """
    rank = comm.Get_rank()

    if rank == 0:
        gen = np.random.RandomState(seed)
        split_seeds = gen.choice(10 * ntot, ntot, replace=False)
    else:
        split_seeds = None

    return comm.bcast(split_seeds, root=0)


###############################################################################
#                         Main fitting functions                              #
###############################################################################


def fit_model(args, frame, pipe, split_seed, tempdir, tempfile_tag):
    """
    Fit the model to the data and calculate the performance metrics and
    feature importances.

    Parameters
    ----------
    args : Namespace
        The command line arguments.
    frame : SPHINXData
        The data frame.
    pipe : sklearn.pipeline.Pipeline
        The regressor pipeline.
    split_seed : int
        The random seed for the train-test split.
    tempdir : str
        The temporary directory for the results.
    tempfile_tag : str
        The tag for the temporary files.

    Returns
    -------
    None
    """
    # Load the X and y data and fit the model.
    Xtrain, Xtest, ytrain, ytest, __, __, __ = frame.make_Xy(
            setup.test_fraction, args.target, args.kind, setup.fesc_min,
            split_seed)
    pipe.fit(Xtrain, ytrain)

    # Score the model
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

    # Calculate feature and permutation importances
    try:
        est = pipe.named_steps["estimator"]
        if args.target == "fesc":
            fimp = est.regressor_.feature_importances_
        else:
            fimp = est.feature_importances_
    except AttributeError:
        fimp = np.full(Xtrain.shape[1], np.nan)
    perm = permutation_importance(
        pipe, Xtest, ytest, n_repeats=setup.perm_repeats,
        random_state=split_seed, n_jobs=1)

    fname_temp = join(
        tempdir,
        f"temp_{args.model}_{args.target}_{tempfile_tag}_{split_seed}.npz")
    np.savez(fname_temp, train_MAE=train_MAE, test_MAE=test_MAE,
             train_R2=train_R2, test_R2=test_R2,
             train_rho=train_rho, test_rho=test_rho,
             feature_importances=fimp,
             permutation_importances=perm.importances_mean,
             permutation_importances_std=perm.importances_std)


###############################################################################
#                           Combine the results                               #
###############################################################################


def combine_results(basepath, tempdir, model, target, kind, tempfile_tag,
                    feature_names):
    """
    Combine the results from the temporary files and save them to a single HDF5
    file.

    Parameters
    ----------
    basepath : str
        The base path for the project.
    tempdir : str
        The temporary directory for the results.
    model : str
        The model name.
    target : str
        The target variable.
    kind : str
        Data set kind.
    tempfile_tag : str
        The tag for the temporary files.
    feature_names : 1-dimensional array
        The feature names.

    Returns
    -------
    fname : str
        The filename of the combined results.
    """
    files = glob(join(tempdir, f"temp_{model}_{target}_{tempfile_tag}_*.npz"))

    nsplit = len(files)
    train_MAE, test_MAE = np.zeros(nsplit), np.zeros(nsplit)
    train_R2, test_R2 = np.zeros(nsplit), np.zeros(nsplit)
    train_rho, test_rho = np.zeros(nsplit), np.zeros(nsplit)

    for n, fname in tqdm(enumerate(files), desc="Loading results"):
        data = np.load(fname)
        train_MAE[n], test_MAE[n] = data["train_MAE"], data["test_MAE"]
        train_R2[n], test_R2[n] = data["train_R2"], data["test_R2"]
        train_rho[n], test_rho[n] = data["train_rho"], data["test_rho"]

        if n == 0:
            shape = (nsplit, len(data["feature_importances"]))
            feature_importances = np.zeros(shape)
            permutation_importance = np.zeros(shape)
            permutation_importance_std = np.zeros(shape)

        feature_importances[n] = data["feature_importances"]
        permutation_importance[n] = data["permutation_importances"]
        permutation_importance_std[n] = data["permutation_importances_std"]

    # Save the combined results
    fname = join(basepath, "results", f"results_{kind}_{model}_{target}.hdf5")
    print(f"{t()}: saving combined results to `{fname}`...", flush=True)
    with File(fname, 'w') as f:
        f.create_dataset("train_MAE", data=train_MAE)
        f.create_dataset("test_MAE", data=test_MAE)
        f.create_dataset("train_R2", data=train_R2)
        f.create_dataset("test_R2", data=test_R2)
        f.create_dataset("train_rho", data=train_rho)
        f.create_dataset("test_rho", data=test_rho)
        f.create_dataset("feature_importances", data=feature_importances)
        f.create_dataset("permutation_importance", data=permutation_importance)
        f.create_dataset("permutation_importance_std",
                         data=permutation_importance_std)

        f.create_dataset("feature_names", data=feature_names)

    return fname


###############################################################################
#                         Commnand line interface                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser(description="Run hyperparameter optimization for scikit-learn models.")  # noqa
    parser.add_argument("--model", type=str, required=True,
                        choices=["ET", "linear"],
                        help="The regressor model to optimize.")
    parser.add_argument("--target", type=str, required=True,
                        choices=["fesc", "nion"],
                        help="The target variable to predict.")
    parser.add_argument("--kind", type=str, required=True,
                        choices=["ALL", "JADES"], help="Dataset to use.")
    parser.add_argument("--nrepeat", type=int, required=True,
                        help="Number of times to repeat the test-train split.")
    parser.add_argument("--machine", type=str, required=True,
                        help="The machine name to get paths.")
    parser.add_argument("--splits_generator_seed", type=int, default=42,
                        help="Random seed for the train-test split.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Get filepaths and the tag of the temporary files.
    basepath = setup.get_basepath(args.machine)
    tempdir = join(basepath, "results", "temp")
    if rank == 0:
        tempfile_tag = str(time())
    else:
        tempfile_tag = None
    tempfile_tag = comm.bcast(tempfile_tag, root=0)

    # Load in the data
    if rank == 0:
        print(f"{t()}: loading data and the regressor...", flush=True)
    frame = photonion.SPHINXData(join(basepath, "data", "all_basic_data.csv"))

    # Prepare the regressor including the best fit hyperparameters
    pipe = setup.load_best_model(args.machine, args.model, args.target,
                                 args.kind)

    # Prepare the random seeds for the train-test splits
    split_seeds = get_split_seeds(args.splits_generator_seed, args.nrepeat,
                                  comm)

    # Create the temporary directory if it does not exist.
    if rank == 0:
        if not isdir(tempdir):
            makedirs(tempdir)

    def fit_model_wargs(i):
        split_seed = split_seeds[i]
        fit_model(args, frame, pipe, split_seed, tempdir, tempfile_tag)

    comm.Barrier()
    work_delegation(fit_model_wargs, list(range(args.nrepeat)), comm,
                    master_verbose=True)
    comm.Barrier()

    # Read off what are the feature names, simplify this?
    feature_names = frame.make_Xy(setup.test_fraction, args.target, args.kind,
                                  setup.fesc_min, 0)[-1]

    if rank == 0:
        fname = combine_results(basepath, tempdir, args.model, args.target,
                                args.kind, tempfile_tag, feature_names)

        with File(fname, 'r') as f:
            # Load back the summary various metrics
            train_MAE_mean, train_MAE_std = mean_std(f["train_MAE"][:])
            test_MAE_mean, test_MAE_std = mean_std(f["test_MAE"][:])
            train_R2_mean, train_R2_std = mean_std(f["train_R2"][:])
            test_R2_mean, test_R2_std = mean_std(f["test_R2"][:])
            train_rho_mean, train_rho_std = mean_std(f["train_rho"][:])
            test_rho_mean, test_rho_std = mean_std(f["test_rho"][:])

            # Load the feature importances and ensure normalization
            fimp_mean, fimp_std = mean_std(f["feature_importances"][:], axis=0)
            fimp_norm = np.sum(fimp_mean)
            fimp_mean /= fimp_norm
            fimp_std /= fimp_norm

            # Load the permutation importance and ensure normalization
            pimp_mean, pimp_std = mean_std(
                f["permutation_importance"][:], axis=0)
            pimp_norm = np.sum(pimp_mean)
            pimp_mean /= pimp_norm
            pimp_std /= pimp_norm

            feature_names = f["feature_names"][:]

        def summarize():
            print()
            print(f"--------------- Summary of the best {args.model} model for {args.kind} ---------------")  # noqa
            print(f"Train log-log MAE   {train_MAE_mean:.5f} +- {train_MAE_std:.5f}")     # noqa
            print(f"Test log-log MAE    {test_MAE_mean:.5f} +- {test_MAE_std:.5f}")       # noqa
            print(f"Train log-log R2    {train_R2_mean:.5f} +- {train_R2_std:.5f}")       # noqa
            print(f"Test log-log R2     {test_R2_mean:.5f} +- {test_R2_std:.5f}")         # noqa
            print(f"Train Spearman      {train_rho_mean:.5f} +- {train_rho_std:.5f}")     # noqa
            print(f"Test Spearman       {test_rho_mean:.5f} +- {test_rho_std:.5f}")       # noqa

            print()

            print("Feature importances:")
            if np.isfinite(fimp_norm):
                for i in range(len(feature_names)):
                    feature_name = str(feature_names[i].decode())
                    print(f"    {feature_name:<10} {fimp_mean[i]:.5f} +- {fimp_std[i]:.5f}")  # noqa

                print()

            print("Permutation importances:")
            for i in range(len(feature_names)):
                feature_name = str(feature_names[i].decode())
                print(f"    {feature_name:<10} {pimp_mean[i]:.5f} +- {pimp_std[i]:.5f}")  # noqa

            print(flush=True)

        summarize()

        with open(fname.replace(".hdf5", ".txt"), 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            summarize()
            sys.stdout = original_stdout

        print(f"{t()}: all finished!", flush=True)
