import os

import numpy as np
from sklearn.linear_model import lars_path


class StabilityLars:
    def __init__(self, nsurrogates=100, mode=1, nTE=1, key=1, maxiterfactor=0.5):
        self.nsurrogates = nsurrogates
        self.mode = mode
        self.nTE = nTE
        self.key = key
        self.maxiterfactor = maxiterfactor

    def _subsampling(self):
        if "mode" in os.environ.keys():  # only for testing
            np.random.seed(200)
        # Subsampling for Stability Selection
        if self.mode == 1:  # different time points are selected across echoes
            subsample_idx = np.sort(
                np.random.choice(range(self.nscans), int(0.6 * self.nscans), 0)
            )  # 60% of timepoints are kept
            if self.nTE > 1:
                for i in range(self.nTE - 1):
                    subsample_idx = np.concatenate(
                        (
                            subsample_idx,
                            np.sort(
                                np.random.choice(
                                    range((i + 1) * self.nscans, (i + 2) * self.nscans),
                                    int(0.6 * self.nscans),
                                    0,
                                )
                            ),
                        )
                    )
        elif self.mode > 1:  # same time points are selected across echoes
            subsample_idx = np.sort(
                np.random.choice(range(self.nscans), int(0.6 * self.nscans), 0)
            )  # 60% of timepoints are kept

        return subsample_idx

    def stability_lars(self, X, Y):

        self.nscans = X.shape[1]

        nvoxels = Y.shape[1]
        nlambdas = self.nscans + 1

        self.auc = np.empty((self.nscans, nvoxels))

        for vox_idx in range(nvoxels):
            lambdas = np.zeros((self.nsurrogates, nlambdas), dtype=np.float32)
            coef_path = np.zeros((self.nsurrogates, self.nscans, nlambdas), dtype=np.float32)
            self.sur_idxs = np.zeros((self.nsurrogates, int(0.6 * self.nscans)))
            for surrogate_idx in range(self.nsurrogates):

                idxs = self._subsampling()
                self.sur_idxs[surrogate_idx, :] = idxs
                y_sub = Y[idxs, vox_idx]
                X_sub = X[idxs, :]
                # max_lambda = abs(np.dot(X_sub.T, y_sub)).max()
                from pywt import wavedec
                from scipy.stats import median_absolute_deviation

                _, cD1 = wavedec(y_sub, "db6", level=1, axis=0)
                lambda_min = median_absolute_deviation(cD1) / 0.6745
                lambda_min = lambda_min / y_sub.shape[0]
                # lambda_min = 0

                # LARS path
                lambdas_temp, _, coef_path_temp = lars_path(
                    X_sub,
                    np.squeeze(y_sub),
                    method="lasso",
                    Gram=np.dot(X_sub.T, X_sub),
                    Xy=np.dot(X_sub.T, np.squeeze(y_sub)),
                    max_iter=self.nscans,  # int(np.ceil(self.maxiterfactor * self.nscans)),
                    eps=1e-6,
                    alpha_min=lambda_min,
                )
                lambdas[surrogate_idx, : len(lambdas_temp)] = lambdas_temp
                n_coefs = (coef_path_temp != 0).shape[1]
                coef_path[surrogate_idx, :, :n_coefs] = coef_path_temp != 0

            # Sorting and getting indexes
            lambdas_merged = lambdas.copy()
            lambdas_merged = lambdas_merged.reshape((nlambdas * self.nsurrogates,))
            sort_idxs = np.argsort(-lambdas_merged)
            lambdas_merged = -np.sort(-lambdas_merged)
            nlambdas_merged = len(lambdas_merged)

            temp = np.zeros((self.nscans, self.nsurrogates * nlambdas), dtype=np.float64)

            for surrogate_idx in range(self.nsurrogates):
                if surrogate_idx == 0:
                    first = 0
                    last = nlambdas - 1
                else:
                    first = last + 1
                    last = first + nlambdas - 1

                same_lambda_idxs = np.where((first <= sort_idxs) & (sort_idxs <= last))[0]

                # Find indexes of changes in value (0 to 1 changes are expected).
                # nonzero_change_scans, nonzero_change_idxs =
                # np.where(np.squeeze(coef_path[surrogate_idx, :, :-1]) !=
                # np.squeeze(coef_path[surrogate_idx, :, 1:]))
                coef_path_temp = np.squeeze(coef_path[surrogate_idx, :, :])
                if len(coef_path_temp.shape) == 1:
                    coef_path_temp = coef_path_temp[:, np.newaxis]
                diff = np.diff(coef_path_temp)
                nonzero_change_scans, nonzero_change_idxs = np.where(diff)
                nonzero_change_idxs = nonzero_change_idxs + 1

                # print(f'{nonzero_change_idxs}')

                coef_path_squeezed = np.squeeze(coef_path[surrogate_idx, :, :])
                coef_path_merged = np.full(
                    (self.nscans, nlambdas * self.nsurrogates), False, dtype=bool
                )
                coef_path_merged[:, same_lambda_idxs] = coef_path_squeezed.copy()

                for i in range(len(nonzero_change_idxs)):
                    coef_path_merged[
                        nonzero_change_scans[i],
                        same_lambda_idxs[nonzero_change_idxs[i]] :,
                    ] = True

                # Sum of non-zero coefficients
                temp += coef_path_merged

            auc_temp = np.zeros((self.nscans,), dtype=np.float64)
            lambda_sum = np.sum(lambdas_merged)

            for lambda_idx in range(nlambdas_merged):
                auc_temp += (
                    temp[:, lambda_idx]
                    / self.nsurrogates
                    * lambdas_merged[lambda_idx]
                    / lambda_sum
                )

            # Saves demeaned AUC
            # demeaned_auc = auc_temp-np.mean(auc_temp)
            # demeaned_auc[demeaned_auc < 0] = 0
            demeaned_auc = auc_temp.copy()

            self.auc[:, vox_idx] = np.squeeze(demeaned_auc)
