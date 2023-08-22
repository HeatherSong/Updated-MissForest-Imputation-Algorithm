from __future__ import division
from functools import partial
import itertools

import numpy as np
from scipy.spatial import distance
from scipy.sparse import issparse

from sklearn.metrics.pairwise import _VALID_METRICS, _return_float_dtype
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.metrics.pairwise import _parallel_pairwise
from sklearn.utils import check_array

def masked_euclidean_distances(X, Y=None, squared=False,
                               missing_values="NaN", copy=True):

    # NOTE: force_all_finite=False allows not only NaN but also +/- inf
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
                                 force_all_finite=False, copy=copy)
    if (np.any(np.isinf(X)) or
            (Y is not X and np.any(np.isinf(Y)))):
        raise ValueError(
            "+/- Infinite values are not allowed.")

    # Get missing mask for X and Y.T
    mask_X = _get_mask(X, missing_values)

    YT = Y.T
    mask_YT = mask_X.T if Y is X else _get_mask(YT, missing_values)

    # Check if any rows have only missing value
    if np.any(mask_X.sum(axis=1) == X.shape[1])\
            or (Y is not X and np.any(mask_YT.sum(axis=0) == Y.shape[1])):
        raise ValueError("One or more rows only contain missing values.")

    # else:
    if missing_values not in ["NaN", np.nan] and (
            np.any(np.isnan(X)) or (Y is not X and np.any(np.isnan(Y)))):
        raise ValueError(
            "NaN values present but missing_value = {0}".format(
                missing_values))

    # Get mask of non-missing values set Y.T's missing to zero.
    # Further, casting the mask to int to be used in formula later.
    not_YT = (~mask_YT).astype(np.int32)
    YT[mask_YT] = 0

    # Get X's mask of non-missing values and set X's missing to zero
    not_X = (~mask_X).astype(np.int32)
    X[mask_X] = 0

    # Calculate distances
    # The following formula derived by:
    # Shreya Bhattarai <shreya.bhattarai@gmail.com>

    distances = (
            (X.shape[1] / (np.dot(not_X, not_YT))) *
            (np.dot(X * X, not_YT) - 2 * (np.dot(X, YT)) +
             np.dot(not_X, YT * YT)))

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)

_MASKED_METRICS = ['masked_euclidean']
_VALID_METRICS += ['masked_euclidean']


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


def check_pairwise_arrays(X, Y, precomputed=False, dtype=None,
                          accept_sparse='csr', force_all_finite=True,
                          copy=False):
    
    X, Y, dtype_float = _return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                            copy=copy, force_all_finite=force_all_finite,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse=accept_sparse, dtype=dtype,
                        copy=copy, force_all_finite=force_all_finite,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y


def _pairwise_callable(X, Y, metric, **kwds):
    force_all_finite = False if callable(metric) else True
    X, Y = check_pairwise_arrays(X, Y, force_all_finite=force_all_finite)

    if X is Y:
        # Only calculate metric for upper triangle
        out = np.zeros((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

        # Make symmetric
        # NB: out += out.T will produce incorrect results
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = X[i]
            out[i, i] = metric(x, x, **kwds)

    else:
        # Calculate all cells
        out = np.empty((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

    return out


# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS['masked_euclidean'] = masked_euclidean_distances


def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=1, **kwds):
    
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    if metric in _MASKED_METRICS or callable(metric):
        missing_values = kwds.get("missing_values") if kwds.get(
            "missing_values") is not None else np.nan

        if np.all(_get_mask(X.data if issparse(X) else X, missing_values)):
            raise ValueError(
                "One or more samples(s) only have missing values.")

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not"
                            " support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        X, Y = check_pairwise_arrays(X, Y, dtype=dtype)

        if n_jobs == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric,
                                                      **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)


import warnings

import numpy as np
from scipy.stats import mode

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

__all__ = [
    'MissForest',
]


class MissForest(BaseEstimator, TransformerMixin):

    def __init__(self, max_iter=3, decreasing=False, missing_values=np.nan,
                 copy=True, n_estimators=70, criterion=('squared_error', 'gini'),
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None):

        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def _miss_forest(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

            # Reg criterion
            reg_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[0]

            # Instantiate regression model
            rf_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=reg_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

            # Classfication criterion
            clf_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[1]

            # Instantiate classification model
            rf_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=clf_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                class_weight=self.class_weight)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        while (
                gamma_new < gamma_old or gamma_newcat < gamma_oldcat) and \
                self.iter_count_ < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_])) / n_catmissing
            if self.num_vars_ is not None:
                gamma_new = np.sum((Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            print("Iteration:", self.iter_count_)
            self.iter_count_ += 1

        return Ximp_old

    def fit(self, X, y=None, cat_vars=None):

        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True

        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array "
                        "of ints.")
            else:
                raise ValueError("cat_vars needs to be either an int or an array "
                                 "of ints.")

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars is not None else None
        col_modes = mode(
            X[:, cat_vars], axis=0, nan_policy='omit')[0] if cat_vars is not \
                                                           None else None

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def transform(self, X):
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_)) \
            + (0 if self.cat_vars_ is None else len(self.cat_vars_))
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")

        # Check if anything is actually missing and if not return original X                             
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original "
                          "dataset.")
            return X

        # row_total_missing = mask.sum(axis=1)
        # if not np.any(row_total_missing):
        #     return X

        # Call missForest function to impute missing
        X = self._miss_forest(X, mask)

        # Return imputed dataset
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)

