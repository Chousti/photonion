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
ML pipeline for predicting LyC (escaping) luminosity or escape fractions.
"""
import sys
from io import StringIO
from os.path import join
from warnings import catch_warnings, simplefilter

import numpy as np
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.utils import Uniform, load_nde_sbi
from joblib import dump, load
from scipy.optimize import minimize
from scipy.stats import kstest, norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import Tensor
from sklearn.neighbors import KernelDensity
from tqdm import trange
from yaml import safe_load

from .io import JADES_filters,feature_set

###############################################################################
#                          Various transforms                                 #
###############################################################################


def KL_loss_fraction(y_true, y_pred):
    """
    Get the "Kullback-Leibler divergence" loss function for the escape
    fractions, which are bounded between 0 and 1.

    Parameters
    ----------
    y_true : 1-dimensional array
        The true values.
    y_pred : 1-dimensional array
        The predicted values.

    Returns
    -------
    loss : float
    """
    return np.mean(
        y_true * np.log(y_true / y_pred)
        + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))
        )


def MAE_log_log(y_true, y_pred, median=False):
    """
    Get the mean (median) absolute error (MAE) between the logarithms of the
    true and predicted values.

    Parameters
    ----------
    y_true : 1-dimensional array
        The true values.
    y_pred : 1-dimensional array
        The predicted values.
    median : bool, optional
        Whether to calculate the median instead of mean.

    Returns
    -------
    loss : float
    """
    if median:
        return np.median(np.abs(np.log10(y_true / y_pred)))
    return np.mean(np.abs(np.log10(y_true / y_pred)))


def RMSE_log_log(y_true, y_pred):
    """
    Get the root mean squared error (RMSE) between the logarithms of the true
    and predicted values.

    Parameters
    ----------
    y_true : 1-dimensional array
        The true values.
    y_pred : 1-dimensional array
        The predicted values.

    Returns
    -------
    loss : float
    """
    return np.sqrt(np.mean((np.log10(y_true / y_pred))**2))


def logit_transform(y, alpha=1):
    """
    Logit transformation of the target variable.

    Parameters
    ----------
    y : array-like
        The target variable.
    alpha : float, optional
        A parameter to control the transform, by default 1 (logit).

    Returns
    -------
    array-like
    """
    y_adj = np.clip(y, 1e-15, 1 - 1e-15)
    return np.log(y_adj**alpha / (1 - y_adj**alpha))


def inverse_logit_transform(y, alpha=1):
    """
    Inverse logit transformation of the target variable.

    Parameters
    ----------
    y : array-like
        The target variable.
    alpha : float, optional
        A parameter to control the transform, by default 1 (logit).

    Returns
    -------
    array-like
    """
    return 1 / (1 + np.exp(-y))**(1 / alpha)


def fit_logit_transform(y):
    """
    Fit the `alpha` parameter for the logit transformation such that the
    transformed variable is normally distributed.

    Parameters
    ----------
    y : 1-dimensional array
        The target variable.

    Returns
    -------
    `scipy.optimize.OptimizeResult`
    """
    def loss(alpha):
        ytf = logit_transform(y, alpha)
        ps = norm.fit(ytf)
        return -np.log(kstest(ytf, norm(*ps).cdf).pvalue)

    bounds = [(0.00001, 3.0)]
    return minimize(loss, x0=0.5, method="Nelder-Mead", bounds=bounds)


###############################################################################
#                      Basic scikit-learn regressor                           #
###############################################################################


def make_regressor_with_target_logit(regressor, alpha=1):
    """
    Make a regressor with target logit transformation.

    Parameters
    ----------
    regressor : object
        Regressor object from scikit-learn.
    alpha : float, optional
        Parameter for the logit transformation.

    Returns
    -------
    TransformedTargetRegressor
    """
    forward_transform = lambda y: logit_transform(y, alpha)          # noqa
    inverse_transform = lambda y: inverse_logit_transform(y, alpha)  # noqa
    scaler = FunctionTransformer(func=forward_transform,
                                 inverse_func=inverse_transform)
    return TransformedTargetRegressor(regressor=regressor, transformer=scaler)


def make_pipeline(regressor, include_target_logit, feature_scaler="standard",
                  alpha=1):
    """
    Make a pipeline with target logit transformation.

    Parameters
    ----------
    regressor : object
        Regressor object from scikit-learn.
    include_target_logit : bool
        Whether to include the target logit transformation.
    feature_scaler : str or object, optional
        Feature scaler object from scikit-learn.  If "standard", use a
        StandardScaler.
    alpha : float, optional
        Parameter for the logit transformation.

    Returns
    -------
    pipeline : :py:class:`sklearn.pipeline.Pipeline`
        An unfitted sklearn pipeline.
    """
    if include_target_logit:
        reg = make_regressor_with_target_logit(regressor, alpha)
    else:
        reg = regressor

    if feature_scaler == "standard":
        feature_scaler = StandardScaler()

    return Pipeline([("feature_scaler", feature_scaler),
                     ("estimator", reg)]
                    )


###############################################################################
#                             SBI regressor                                   #
###############################################################################


class SBIRegressor:
    """
    SBI regressor for the escape fractions from photometric data built on top
    of the `LtU ILI` package [1]. Employs a mixture density network (MDN) to
    model the neural posterior estimator (NPE) with a uniform prior in
    the scaled target space.

    Applies a standard scaling to the features and either logit + standard or
    only standard scaling to the target variable.

    Parameters
    ----------
    target_scaling : str
        Kind of transformations to apply to the target variable. Must be one
        of: "logit_standard", "standard".
    num_nets : int, optional
        The number of NPEs to use.
    train_args : dict, optional
        The arguments for training the NPE. Options include:
        `training_batch_size`, `learning_rate`, `validation_fraction`,
        `stop_after_epochs`, and `clip_max_norm`.
    model_args : dict, optional
        The arguments for the MDN. Options include:
        `hidden_features` and `num_components`.
    out_dir : str, optional
        The output directory to store the trained model.
    name : str, optional
        Name of the model, used when saving it.
    device : str, optional
        The device to use for training, either "cpu" or "gpu".

    References
    ----------
    [1] https://arxiv.org/abs/2402.05137
    """
    _target_scaling = None
    _device = "cpu"

    def __init__(self, target_scaling, num_nets=1, train_args=None,
                 model_args=None, out_dir=None, name=None, device="cpu", model_type='mdn'):
        self.target_scaling = target_scaling
        self.num_nets = num_nets

        if train_args is not None and not isinstance(train_args, dict):
            raise ValueError("`train_args` must be a dictionary.")
        if model_args is not None and not isinstance(model_args, dict):
            raise ValueError("`model_args` must be a dictionary.")

        # Trainer arguments.
        default_train_args = {"training_batch_size": 64,
                              "learning_rate": 1e-4,
                              "validation_fraction": 0.2,
                              "stop_after_epochs": 15,
                              "clip_max_norm": 5}
        if train_args is None:
            train_args = default_train_args

        if not isinstance(train_args, dict):
            raise ValueError("`train_args` must be a dictionary.")
        # Check for unexpected keys.
        for key in train_args:
            if key not in default_train_args:
                raise ValueError(f"Unexpected key in `train_args`: {key}. "
                                 f"Options are: {list(default_train_args.keys())}.")  # noqa

        for key in default_train_args:
            if key not in train_args:
                train_args[key] = default_train_args[key]
        self._train_args = train_args

        # Model arguments.
        self._model_name = model_type
        if self._model_name == 'mdn':
            default_model_args = {"hidden_features": 50,
                                  "num_components": 4}
        elif self._model_name == 'maf':
            default_model_args = {"hidden_features": 50,
                                  "num_transforms": 4}
        else:
            raise ValueError(f'Unknown model {self._model_name}, currently set up for [mdn, maf]')
        
        if model_args is None:
            model_args = default_model_args
        if not isinstance(model_args, dict):
            raise ValueError("`model_args` must be a dictionary.")
        # Check for unexpected keys.
        for key in model_args:
            if key not in default_model_args:
                raise ValueError(f"Unexpected key in `model_args`: {key}. "
                                 f"Options are: {list(default_model_args.keys())}.")  # noqa

        for key in default_model_args:
            if key not in model_args:
                model_args[key] = default_model_args[key]
        self._model_args = model_args

        self._posterior, self._stats = None, None
        self._feature_scaler = StandardScaler()
        self._target_scaler = StandardScaler()
        self._out_dir = out_dir

        self._name = name + "_" if name is not None else None
        self._from_config = False
        self._alpha = None
        self._device = device

    @property
    def target_scaling(self):
        """
        Kind of transformations to apply to the target variable.

        Returns
        -------
        str
        """
        if self._target_scaling is None:
            raise RuntimeError("`target_scaling` is not set.")
        return self._target_scaling

    @target_scaling.setter
    def target_scaling(self, value):
        options = ["logit_standard", "standard"]
        if value not in options:
            raise ValueError(f"`target_scaling` must be one of: {options}.")
        self._target_scaling = value

    @property
    def num_nets(self):
        """
        Return the number of NPEs.

        Returns
        -------
        int
        """
        if self._num_nets is None:
            raise RuntimeError("`num_nets` is not set.")
        return self._num_nets

    @num_nets.setter
    def num_nets(self, value):
        if not (isinstance(value, int) and value >= 1):
            raise ValueError("`num_nets` must be an integer >= 1.")
        self._num_nets = value

    def transform_target(self, y):
        """
        Transform the target variable.

        Parameters
        ----------
        y : 1-dimensional array
            The target variable.

        Returns
        -------
        1-dimensional array
        """
        if "logit" in self.target_scaling:
            y_scaled = logit_transform(y, self._alpha)
        else:
            y_scaled = y

        mean, scale = self._target_scaler.mean_, self._target_scaler.scale_
        if mean.size > 1 or scale.size > 1:
            raise RuntimeError("Unexpected mean or scale size.")
        mean, scale = mean[0], scale[0]

        return (y_scaled - mean) / scale

    def inverse_transform_target(self, y):
        """
        Inverse transform the target variable.

        Parameters
        ----------
        y : 1-dimensional array
            The target variable.

        Returns
        -------
        1-dimensional array
        """
        mean, scale = self._target_scaler.mean_, self._target_scaler.scale_
        if mean.size > 1 or scale.size > 1:
            raise RuntimeError("Unexpected mean or scale size.")
        mean, scale = mean[0], scale[0]

        if "logit" in self.target_scaling:
            return inverse_logit_transform(y * scale + mean, self._alpha)

        return y * scale + mean

    def fit(self, X, y, verbose=True):
        """
        Fit the regressor.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (n_samples,)
            Target variable.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        None
        """
        if self._from_config:
            raise RuntimeError("Regressors loaded from config cannot be fitted.")  # noqa

        if y.ndim == 1:
            y = np.copy(y).reshape(-1, 1)

        # First scale the features.
        X_scaled = self._feature_scaler.fit_transform(X)

        # Optinally fit the logit transform.
        if "logit" in self.target_scaling:
            res = fit_logit_transform(y)
            if not res.success:
                raise RuntimeError("The logit transformation fit failed.")
            self._alpha = res.x[0]

        if verbose and 'logit' in self.target_scaling:
            print(f"Logit transformation fit: alpha = {self._alpha:.3f}",
                  flush=True)

        if "logit" in self.target_scaling:
            y_scaled = logit_transform(y, self._alpha)
        else:
            y_scaled = y
        y_scaled = self._target_scaler.fit_transform(y_scaled.reshape(-1, 1))
        y_scaled = y_scaled.reshape(-1)

        ystd = y_scaled.std()
        prior = Uniform(low=[np.min(y_scaled) - 3 * ystd],
                        high=[np.max(y_scaled) + 3 * ystd])

        nets = [load_nde_sbi(engine='NPE', model=self._model_name, **self._model_args)  # noqa
                for _ in range(self.num_nets)]

        trainer = InferenceRunner.load(
            backend='sbi', engine='NPE', prior=prior, nets=nets,
            train_args=self._train_args, out_dir=self._out_dir,
            name=self._name,)

        loader = NumpyLoader(X_scaled, y_scaled)

        try:
            if not verbose:
                buffer_stdout = StringIO()
                original_stdout = sys.stdout
                sys.stdout = buffer_stdout

            with catch_warnings():
                simplefilter("ignore", UserWarning)
                self._posterior, self._stats = trainer(loader)
        finally:
            if not verbose:
                sys.stdout = original_stdout

        if self._out_dir is not None and self._name is not None:
            fname = join(self._out_dir, self._name + "aux.pkl")
            print(f"\nSaving additional information to `{fname}`.")
            data = {"feature_scaler": self._feature_scaler,
                    "target_scaler": self._target_scaler,
                    "alpha": self._alpha,
                    "target_scaling": self.target_scaling,
                    "num_nets": self.num_nets,
                    }
            dump(data, fname)

    def sample(self, X, n_samples=1000, inverse_transform=True, verbose=True):
        """
        Produce samples from the learnt posterior distribution.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        n_samples : int, optional
            The number of samples to produce per each input.
        inverse_transform : bool, optional
            Whether to apply the inverse transformations to the target
            variable.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        y_samples : 2-dimensional array of shape (n_samples_input, n_samples)
            Samples from the posterior distribution.
        """
        if self._posterior is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        X_scaled = self._feature_scaler.transform(X)

        samples = np.empty((len(X), n_samples), dtype=np.float32)
        for i in trange(len(X), desc="Sampling", disable=not verbose):
            samples[i] = self._posterior.sample(
                x=X_scaled[i], sample_shape=(n_samples,),
                show_progress_bars=False).reshape(-1,)

        if inverse_transform:
            samples = self.inverse_transform_target(samples)

        return samples

    def sample_summarized(self, X, n_samples=1000, inverse_transform=True,
                          return_samples=False, verbose=True):
        """
        Produce summarized samples, i.e. the mode and the 68% credible interval
        around it.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        n_samples : int, optional
            The number of samples to produce per each input to estimate the
            credible interval.
        inverse_transform : bool, optional
            Whether to apply the inverse transformations to the target
            variable upon outpu.
        return_samples : bool, optional
            Whether to return the samples as well.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        ys_summary : 2-dimensional array of shape (n_samples, 3)
            The mode and the 68% credible interval around it.
        (optional) ys : 2-dimensional array of shape (n_input, n_samples)
            Samples from the learnt posterior distribution.
        """
        ys = self.sample(X, n_samples=n_samples, inverse_transform=False,
                         verbose=verbose)
        ys_summary = self.summarize_samples(ys, verbose=verbose)

        if inverse_transform:
            ys_summary = self.inverse_transform_target(ys_summary)
            ys = self.inverse_transform_target(ys)

        if return_samples:
            return ys_summary, ys

        return ys_summary

    def summarize_samples(self, ys, verbose=True):
        """
        Summarize samples from the posterior distribution by estimating the
        mode and 68% credible interval around it.

        Parameters
        ----------
        ys : 1- or 2-dimensional array of shape (n_input, n_samples)
            Samples from the posterior distribution.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        y : 1- or 2-dimensional array
            Mode of the distribution, upper and lower limit.
        """
        ndim = ys.ndim
        if ys.ndim == 1:
            ys = np.copy(ys).reshape(1, -1)
            verbose = False

        dp = 0.3413447460685429
        ys_summary = np.empty((len(ys), 3), dtype=np.float32)

        for i in trange(len(ys), desc="Summarizing", disable=not verbose):
            y = ys[i]
            kde = KernelDensity(bandwidth="scott", kernel="tophat")
            kde.fit(y.reshape(-1, 1))

            xrange = np.linspace(y.min(), y.max(), 1000)
            dx = xrange[1] - xrange[0]
            pdf = np.exp(kde.score_samples(xrange.reshape(-1, 1)))
            cdf = np.cumsum(pdf) * dx

            k = np.argmax(pdf)
            cdf_peak = cdf[k]
            ys_summary[i, 0] = xrange[k]
            ys_summary[i, 1] = xrange[np.abs(cdf - (cdf_peak + dp)).argmin()]
            ys_summary[i, 2] = xrange[np.abs(cdf - (cdf_peak - dp)).argmin()]

        if ndim == 1:
            ys_summary = ys_summary.reshape(-1)

        return ys_summary

    def log_prob(self, X, y, verbose=True):
        """
        Calculate the log-probability of observed samples.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (n_samples,)
            Target variable.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        1-dimensional array
        """
        lp = np.empty(len(X), dtype=np.float32)

        X_scaled = self._feature_scaler.transform(X)
        y_scaled = self.transform_target(y)

        for i in trange(len(X), desc="Log-probability", disable=not verbose):
            x = Tensor(X_scaled[i])
            theta = Tensor([y_scaled[i]])
            lp[i] = self._posterior.log_prob(x=x, theta=theta,
                                             norm_posterior=True)
        return lp

    def calculate_pit(self, X, y, n_samples=1000, verbose=True):
        """
        Calculate the probability integral transform (PIT) for the samples
        produced by the regressor.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (n_samples,)
            Target variable.

        Returns
        -------
        pit : 1-dimensional array
            The PIT values.
        """
        samples = self.sample(X, n_samples=n_samples, inverse_transform=False,
                              verbose=verbose)

        ytf = self.transform_target(y)

        pit = np.empty(len(y))
        for i in range(len(y)):
            pit[i] = np.mean(samples[i] < ytf[i])

        pit = np.sort(pit)
        pit /= pit[-1]

        return pit

    def evaluate_from_samples(self, ypred, ytrue):
        """
        Calculate various metrics from the samples produced by the regressor.

        Parameters
        ----------
        ypred : 2-dimensional array of shape (n_samples_input, n_samples)
            Samples from the posterior distribution.
        ytrue : 1-dimensional array of shape (n_samples_input,)
            Target variable.

        Returns
        -------
        dict
        """
        if ytrue.ndim == 2 and ytrue.shape[1] > 1:
            raise RuntimeError("Unpexpected shape of `ypred`.")
        ytrue = ytrue.reshape(-1,)

        if not ypred.ndim == 2:
            raise ValueError("`ypred` must be a 2-dimensional array.")

        # Transform to the space in which we fitted the regressors.
        ypred_tf = self.transform_target(ypred)
        ytrue_tf = self.transform_target(ytrue)

        ypred_tf_mean = np.mean(ypred_tf, axis=1)

        ypred_mean = self.inverse_transform_target(ypred_tf_mean)

        return {
            "Mean_AE_loglog": MAE_log_log(ytrue, ypred_mean),
            "Median_AE_loglog": MAE_log_log(ytrue, ypred_mean, median=True),
            "RMSE_loglog": RMSE_log_log(ytrue, ypred_mean),
            "Mean_AE_latent": np.mean(np.abs(ytrue_tf - ypred_tf_mean)),
            "Median_AE_latent": np.median(np.abs(ytrue_tf - ypred_tf_mean)),
            "RMSE_latent": np.sqrt(np.mean((ytrue_tf - ypred_tf_mean)**2)),
                }

    def evaluate(self, X, y, n_samples=1000, verbose=True):
        """
        Calculate various metrics from the samples produced by the regressor.

        Parameters
        ----------
        X : 2-dimensional array of shape (n_samples, n_features)
            Feature array.
        y : 1-dimensional array of shape (n_samples,)
            Target variable.
        n_samples : int, optional
            The number of samples to produce per each input.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        dict
        """
        ypred = self.sample(X, n_samples=n_samples, verbose=verbose)
        return self.evaluate_from_samples(ypred, y)

    @property
    def validation_log_probs(self):
        """
        Validation set log-probability of each epoch for each net.

        Returns
        -------
        list of 1-dimensional arrays
        """
        if self._stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["validation_log_probs"] for stat in self._stats]

    @property
    def training_log_probs(self):
        """
        Training set log-probability of each epoch for each net.

        Returns
        -------
        list of 1-dimensional array
        """
        if self._stats is None:
            raise RuntimeError("The regressor has not been fitted yet.")

        return [stat["training_log_probs"] for stat in self._stats]

    @classmethod
    def from_config(cls, out_dir, name):
        """
        Load the regressor from the configuration files.

        Parameters
        ----------
        out_dir : str
            The output directory.
        name : str
            The name of the regressor.
        """
        aux = load(join(out_dir, name + "_aux.pkl"))
        model = cls(target_scaling=aux["target_scaling"], out_dir=out_dir,
                    name=name)
        model._posterior = load(join(out_dir, name + "_posterior.pkl"))

        with open(join(out_dir, name + "_summary.json"), "r") as fd:
            model._stats = safe_load(fd)

        model._feature_scaler = aux["feature_scaler"]
        model._target_scaler = aux["target_scaler"]
        model._alpha = aux["alpha"]
        model.num_nets = aux["num_nets"]

        model._from_config = True
        return model
