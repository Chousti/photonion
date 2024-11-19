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
Functions for reading and writing data.
"""
import warnings
from os.path import exists

import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
cosmology = FlatLambdaCDM(70, 0.3)

JADES_filters = ["F115W", "F150W", "F200W", "F277W", "F335M", "F356W", "F410M",
                 "F444W"]

feature_set = ["F115W", "F150W", "F200W", "F277W", "F335M", "F356W", "F410M", "F444W",'F115W_F150W',
                 'F150W_F277W', 'F277W_F444W', 'MAB']

###############################################################################
#                           Reading SPHINX data                               #
###############################################################################


class SPHINXData:
    """
    Class for conveniently loading and manipulating the SPHINX galaxy
    catalogue from the `all_basic_data.csv` file.

    Parameters
    ----------
    fpath : str
        Path to the `all_basic_data.csv` file.
    """
    def __init__(self, fpath):
        self._data = np.genfromtxt(fpath, skip_header=1, delimiter=",")

        with open(fpath, 'r') as f:
            first_line = f.readline()

        columns = first_line.split(",")
        self._col2indx = {col: i for i, col in enumerate(columns)}

    def _get_directed(self, key):
        """
        Get data that looks like f"{key}_{n}", for example "F115W_dir_0".

        Parameters
        ----------
        key : str
            The key to look for, for example f"F115W_dir_{n}" for n=0, 1, ...

        Returns
        -------
        2-dimensional array of shape (ngals, ndir)
        """
        y, n = [], 0
        while True:
            try:
                y.append(self[f"{key}_{n}"])
            except KeyError:
                break

            n += 1

        return np.asanyarray(y).T

    def make_Xy(self, test_fraction, target_name="fesc", kind="JADES",
                fesc_min=1e-16, seed=42, reweigh_data=False, n_reweigh=10, muv_norm=False):
        """
        Prepare the data for a machine learning model, the features
        are the filter data and the redshift, the target is the escape
        fraction. Each galaxy is observed in multiple directions, and thus
        repeated in the output data multiple times but from different
        directions.

        Parameters
        ----------
        test_fraction : float
            The fraction of the data to be used for testing the model.
        target_name : str, optional
            The target of the model. Can be either "fesc" or "nion".
        kind : str
            Kind of which filters to select. Can be either "JADES" or "ALL".
        fesc_min : float, optional
            Minimum escape fraction to be included in the data.
        seed : int, optional
            Random seed for the train-test split.

        Returns
        -------
        Xtrain, Xtest, ytrain, ytest : 2-dimensional arrays
            The training and testing data for the model.
        indxs_train, indxs_test : 1-dimensional arrays
            The indices of the galaxies in the training and testing data.
        feature_names : list of str
            The names of the features in the data.
        """
        if not 0 < test_fraction < 1:
            raise ValueError("`test_fraction` must be in (0, 1)")

        # First off load the filter data
        if kind== "JADES_filters":
            filter_keys = JADES_filters
        if kind == "JADES":
            filter_keys = ["F115W", "F150W", "F200W", "F277W", "F335M", "F356W", "F410M", "F444W",'F115W_F150W', 'F150W_F277W', 'F277W_F444W', 'MAB']
        elif kind == "ALL":
            filter_keys = ["F115W", "F150W", "F200W", "F277W", "F356W",
                           "F444W", "F140M", "F162M", "F182M", "F210M",
                           "F250M", "F300M", "F335M", "F360M", "F410M",
                           "F430M", "F460M", "F480M"]
        else:
            raise ValueError(f"Invalid kind `{kind}`")

        def check_if_color(s):
            y = s.split('_')
            return (len(y)>1), y
        
        def get_MABs(f090, f115, f150, f200, f277, f356, z, return_muv=False):
            # Function to compute M_AB at rest-1500A by interpolating JWST NIRCam photometric magnitudes

            def func_loc(x,a,b):
                # Interpolating function
                return a*x**b
            
            # Photometric filter wavelengths
            #  90, 115, 150, 200, 277, 335, 356, 410, 444
            ws = np.array([0.89824364, 1.14859202, 1.49442228, 1.97811383, 2.76120868, 3.3587610, 3.54834844, 4.07932817, 4.37878307])
            
            def get_muv(ab,z):
                # function to convert from apparent AB magnitude to absolute UV magnitude (M_uv)
                lum_dist = cosmology.luminosity_distance(z).to('pc').value
                M = ab + 5*(1 - np.log10(lum_dist)) + 2.5*np.log10(1+z)
                return M
            
            MABs = np.zeros((len(z), 10))
            for i in range(len(z)):
                for j in range(10):
                    # Select correct filter combination for redshift
                    if (z[i] >= 4.5) & (z[i] < 5.25):
                        flux = np.array([f090[i,j], f115[i,j], f150[i,j]])
                        w = np.array([ws[0], ws[1], ws[2]])
                    elif (z[i]>= 5.25) & (z[i] < 7.25):
                        flux = np.array([f115[i,j], f150[i,j], f200[i,j]])
                        w = np.array([ws[1], ws[2], ws[3]])
                    elif (z[i] >= 7.25) & (z[i] < 9.75):
                        flux = np.array([f150[i,j], f200[i,j], f277[i,j]])
                        w = np.array([ws[2], ws[3], ws[4]])
                    elif (z[i] >= 9.75) & (z[i] <= 13):
                        flux = np.array([f200[i,j], f277[i,j], f356[i,j]])
                        w = np.array([ws[3], ws[4], ws[6]])
                    else:
                        print(f'{z[i]} outside redshift range, no known model')
                        continue
                    
                    # Fit rough shape of continuum
                    val, pcov = curve_fit(func_loc, w, (flux))
                    
                    # Find value at 1500A (redshifted)
                    lam = 1500  / 1e4 * (1 + z[i])
                    lam2 = 1500 * (1 + z[i]) * 1e-10
                    MAB = func_loc(lam, *val)

                    # Return results
                    if return_muv:
                        MABs[i,j] = get_muv(MAB, z[i])
                    else:
                        MABs[i,j] = MAB
            return MABs

        filter_data = None
        for i, key in enumerate(filter_keys):
            check,key_split = check_if_color(key)
            if check:
                x = self._get_directed(key_split[0] + "_dir") - self._get_directed(key_split[1] + "_dir")
            else:
                if key == 'MAB':
                    x = get_MABs(self._get_directed('F090W_dir'), self._get_directed('F115W_dir'), self._get_directed('F150W_dir'), 
                                self._get_directed('F200W_dir'), self._get_directed('F277W_dir'), self._get_directed('F356W_dir'), 
                                self['redshift']
                                )
                    check = True # so that we don't renormalise by MAB
                else:
                    x = self._get_directed(key + "_dir")
            if filter_data is None:
                filter_data = np.full((*x.shape, len(filter_keys)), np.nan)

            if muv_norm and not check:
                y = get_MABs(self._get_directed('F090W_dir'), self._get_directed('F115W_dir'), self._get_directed('F150W_dir'), 
                                self._get_directed('F200W_dir'), self._get_directed('F277W_dir'), self._get_directed('F356W_dir'), 
                                self['redshift']
                                )
                x = x - y

            filter_data[..., i] = x

        redshift = self["redshift"]
        if target_name == "fesc":
            target = 10**self["f_esc"]
            mask = target > fesc_min
        elif target_name == "nion":
            target = self['f_esc'] + self['ionizing_luminosity']  # log10
            mask = (~np.isnan(target)) & (self['f_esc'] > np.log10(fesc_min))
        else:
            raise ValueError(f"Invalid target name `{target_name}`")

        gal_indxs = np.arange(len(redshift))
        feature_names = filter_keys + ["redshift"]

        redshift = redshift[mask]
        target = target[mask]
        filter_data = filter_data[mask]
        gal_indxs = gal_indxs[mask]

        if reweigh_data:
            if fesc_min < 1e-10:
                bins=np.linspace(48, np.max(target), n_reweigh)
                bins = np.hstack((np.array([42]), bins))
            else:
                bins=np.linspace(np.min(target), np.max(target), n_reweigh+1)
            digits = np.digitize(target, bins)
            for i in range(n_reweigh):
                filter_digits = (digits==i+1)
                if i == 0:
                    train_indxs, test_indxs = train_test_split(np.arange(len(redshift))[filter_digits], test_size=test_fraction,random_state=seed)
                else:
                    temp_train, temp_test = train_test_split(np.arange(len(redshift))[filter_digits], test_size=test_fraction,random_state=seed)
                    train_indxs = np.hstack((train_indxs, temp_train))
                    test_indxs = np.hstack((test_indxs, temp_test))
        else:
            train_indxs, test_indxs = train_test_split(
                np.arange(len(redshift)), test_size=test_fraction,
                random_state=seed)

        # Initialize the X and y arrays
        ndir = filter_data.shape[1]
        ntrain, ntest = len(train_indxs), len(test_indxs)
        Xtrain = np.full((ntrain * ndir, len(filter_keys) + 1), np.nan)
        Xtest = np.full((ntest * ndir, len(filter_keys) + 1), np.nan)
        ytrain = np.full(ntrain * ndir, np.nan)
        ytest = np.full(ntest * ndir, np.nan)
        gal_indxs_train = np.full(ntrain * ndir, -1, dtype=int)
        gal_indxs_test = np.full(ntest * ndir, -1, dtype=int)

        for i in range(ndir):
            itrain, jtrain = i * ntrain, (i + 1) * ntrain
            itest, jtest = i * ntest, (i + 1) * ntest

            # Add filters
            Xtrain[itrain:jtrain, :-1] = filter_data[train_indxs, i, :]
            Xtest[itest:jtest, :-1] = filter_data[test_indxs, i, :]

            # Add redshift to the last column
            Xtrain[itrain:jtrain, -1] = redshift[train_indxs]
            Xtest[itest:jtest, -1] = redshift[test_indxs]

            # Make the y arrays
            ytrain[itrain:jtrain] = target[train_indxs]
            ytest[itest:jtest] = target[test_indxs]

            # Add the galaxy indices
            gal_indxs_train[itrain:jtrain] = gal_indxs[train_indxs]
            gal_indxs_test[itest:jtest] = gal_indxs[test_indxs]

        return (Xtrain, Xtest, ytrain, ytest, gal_indxs_train, gal_indxs_test,
                feature_names)

    def keys(self):
        """
        Keys of the columns in the SPHINX data.

        Parameters
        ----------
        None

        Returns
        -------
        list of str
        """
        return list(self._col2indx.keys())

    def __getitem__(self, key):
        if key not in self._col2indx:
            raise KeyError(f"Column `{key}` not found in data")
        return self._data[:, self._col2indx[key]]
    