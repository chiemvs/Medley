"""
Functions for cross validation.
Completely based on / copied from code at 
https://doi.org/10.5281/zenodo.7967133
from publication:
Sweet, L. B., Müller, C., Anand, M., & Zscheischler, J. (2023). Cross-validation strategy impacts the performance and interpretation of machine learning models. Artificial Intelligence for the Earth Systems, 2(4), e230026.
"""

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.model_selection import KFold, StratifiedKFold, BaseCrossValidator
from sklearn.cluster import MiniBatchKMeans, BisectingKMeans
from sklearn.utils.validation import check_array

from functools import reduce


class SpatiotemporalSplit(_BaseKFold):
    """
    Spatiotemporal cross-validator

    Splits data by time and/or space.
    Time splitting is NOT done such that the test datapoints in each split are
    always later than the training datapoints.

    """

    def __init__(
        self,
        n_temporal=None,
        n_spatial=None,
        *,
        spatial_method=None,
        time_dim=None,
        spatial_dim=None,
    ):
        n_splits = reduce(
            lambda x, y: x * y, [a for a in [n_temporal, n_spatial] if a is not None]
        )
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.spatial_method = spatial_method
        self.n_temporal = n_temporal
        self.n_spatial = n_spatial
        self.time_dim = time_dim
        self.spatial_dim = spatial_dim

    def split(self, X, y=None, groups=None):
        if self.time_dim is None and self.spatial_dim is None:
            raise ValueError(
                "The 'time_dim' and 'spatial_dim' parameter should not both be None."
            )
        if (self.time_dim is not None and self.n_temporal is None) or (
            self.time_dim is None and self.n_temporal is not None
        ):
            raise ValueError(
                "Either neither or both of 'time_dim' and 'n_temporal' should be None."
            )
        if (self.spatial_dim is not None and self.n_spatial is None) or (
            self.spatial_dim is None and self.n_spatial is not None
        ):
            raise ValueError(
                "Either neither or both of 'spatial_dim' and 'n_spatial' should be None."
            )

        indices = np.arange(len(X))

        if self.n_temporal is not None:
            unique_t = np.unique(self.time_dim)

            if self.n_temporal > len(unique_t):
                raise ValueError(
                    "Cannot have number of temporal splits=%d greater"
                    " than the number of unique timesteps: %d."
                    % (self.n_temporal, unique_t)
                )

            time_folds = np.array_split(np.sort(unique_t), self.n_temporal)

            if self.n_spatial is None:
                for time_fold in time_folds:
                    time_indices = np.isin(self.time_dim, time_fold)
                    yield (
                        indices[~time_indices],
                        indices[time_indices],
                    )

        if self.n_spatial is not None:
            if self.spatial_method == "1D":
                unique_s = np.unique(self.spatial_dim)

                if self.n_spatial > len(unique_s):
                    raise ValueError(
                        "Cannot have number of spatial splits=%d greater"
                        " than the number of unique spatial labels: %d."
                        % (self.n_spatial, unique_s)
                    )

                boundaries = np.percentile(
                    self.spatial_dim,
                    (np.arange(self.n_spatial + 1) / self.n_spatial) * 100,
                )

            if self.n_temporal is None:
                for i in np.arange(self.n_spatial):
                    spatial_indices = (self.spatial_dim >= boundaries[i]) & (
                        self.spatial_dim < boundaries[i + 1]
                    )
                    if i == (self.n_spatial - 1):
                        spatial_indices = (self.spatial_dim >= boundaries[i]) & (
                            self.spatial_dim <= boundaries[i + 1]
                        )
                    yield (
                        indices[~spatial_indices],
                        indices[spatial_indices],
                    )

        if self.n_temporal is not None and self.n_spatial is not None:
            for time_fold in time_folds:
                for i in np.arange(self.n_spatial):
                    time_indices = np.isin(self.time_dim, time_fold)
                    spatial_indices = (self.spatial_dim >= boundaries[i]) & (
                        self.spatial_dim < boundaries[i + 1]
                    )
                    if i == (self.n_spatial - 1):
                        spatial_indices = (self.spatial_dim >= boundaries[i]) & (
                            self.spatial_dim <= boundaries[i + 1]
                        )
                    yield (
                        indices[~time_indices & ~spatial_indices],
                        indices[time_indices & spatial_indices],
                    )


