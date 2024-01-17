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


class DimensionalCV(BaseCrossValidator):
    """
    Dimensional cross-validator.

    Splits data by given dimensions into a certain number of folds.

    """

    def __init__(self, dims, n_folds):
        super().__init__()
        self.dims = dims
        self.n_folds = n_folds

    def _iter_test_masks(self, X, y, groups=None):
        if self.dims is None:
            raise ValueError("The 'dims' parameter should not be None.")

        # Define groups according to dims and n_folds
        boundaries = np.percentile(
            self.dims[0],
            (np.arange(self.n_folds[0] + 1) / self.n_folds[0]) * 100,
        )
        groups = np.zeros(self.dims[0].shape)
        for i in np.arange(self.n_folds[0]):
            if i == (self.n_folds[0] - 1):
                groups[
                    (self.dims[0] >= boundaries[i])
                    & (self.dims[0] <= boundaries[i + 1])
                ] = i
            else:
                groups[
                    (self.dims[0] >= boundaries[i]) & (self.dims[0] < boundaries[i + 1])
                ] = i

        if len(self.dims) == 2:
            for i in np.arange(self.n_folds[0]):
                inner_boundaries = np.percentile(
                    self.dims[1][groups == i],
                    (np.arange(self.n_folds[1] + 1) / self.n_folds[1]) * 100,
                )
                for j in np.arange(self.n_folds[1]):
                    if j == (self.n_folds[1] - 1):
                        groups[
                            (self.dims[1] >= inner_boundaries[j])
                            & (self.dims[1] <= inner_boundaries[j + 1])
                            & (groups == i)
                        ] = (i + self.n_folds[0] * j)
                    else:
                        groups[
                            (self.dims[1] >= inner_boundaries[j])
                            & (self.dims[1] < inner_boundaries[j + 1])
                            & (groups == i)
                        ] = (i + self.n_folds[0] * j)

        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if len(unique_groups) != np.prod(self.n_folds):
            raise ValueError(
                f"Could not create as many groups as n_folds specified. Created {len(unique_groups)}."
            )
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.prod(self.n_folds)

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)


class ClusteredCV(DimensionalCV):
    """
    Clustered cross-validator.
    """

    def __init__(self, cluster_dims, n_clusters, random_state=0):
        super().__init__(cluster_dims, n_clusters)
        self.cluster_dims = cluster_dims
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _iter_test_masks(self, X, y=None, groups=None):
        clusters = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        clusters.fit(np.column_stack(self.cluster_dims))
        groups = clusters.labels_
        unique_groups = np.unique(groups)
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.prod(self.n_clusters)

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)


class IterativeClusteredCV(DimensionalCV):
    """
    Clustered cross-validator using BisectingKMeans.
    """

    def __init__(self, cluster_dims, n_clusters, random_state=0):
        super().__init__(cluster_dims, n_clusters)
        self.cluster_dims = cluster_dims
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _iter_test_masks(self, X, y=None, groups=None):
        clusters = BisectingKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            bisecting_strategy="largest_cluster",
        )
        clusters.fit(np.column_stack(self.cluster_dims))
        groups = clusters.labels_
        unique_groups = np.unique(groups)
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        return np.prod(self.n_clusters)

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)


def get_cv(name, x, experiment):
    """
    Given the name of the preferred CV method, split the dataset x accordingly
    and return the name (which may be different, if input wasn't understood).
    Experiment is required to know whether to use Classifier or Regressor.
    """

    if name is None or name == "kfold":
        if experiment == "yield_var":
            cv = KFold(n_splits=20, shuffle=True, random_state=0)
        elif experiment == "yield_failure":
            cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)
        name = "kfold"
        print("Creating 20 random CV folds...")
    elif name == "temporal":
        cv = SpatiotemporalSplit(n_temporal=20, time_dim=x.year)
        print("Creating 20 temporal CV folds...")
    elif name == "latitude":
        cv = SpatiotemporalSplit(
            n_spatial=20,
            spatial_dim=x.lat,
            spatial_method="1D",
        )
        print("Creating 20 latitude CV folds...")
    elif name == "longitude":
        cv = SpatiotemporalSplit(
            n_spatial=20,
            spatial_dim=x.lon,
            spatial_method="1D",
        )
        print("Creating 20 longitude CV folds...")
    elif name == "spatiotemporal":
        cv = SpatiotemporalSplit(
            n_temporal=4,
            time_dim=x.year,
            n_spatial=5,
            spatial_dim=x.lat,
            spatial_method="1D",
        )
        print("Creating 20 CV folds, 3 temporal and 3 by latitude...")
    elif name == "spatial_clusters":
        cv = ClusteredCV(cluster_dims=[x.lat, x.lon], n_clusters=20, random_state=0)
        print("Creating 20 CV folds by spatially clustering...")
    elif name == "spatiotemporal_clusters":
        cv = ClusteredCV(
            cluster_dims=[x.lat, x.lon, x.year], n_clusters=20, random_state=0
        )
        print("Creating 20 CV folds by spatiotemporally clustering...")
    elif name == "feature_clusters":
        cv = IterativeClusteredCV(cluster_dims=[x], n_clusters=20, random_state=0)
        print("Creating 20 CV folds by clustering in feature space...")
    else:
        print("CV input not understood. Creating 20 random CV folds...")
        if experiment == "yield_var":
            cv = KFold(n_splits=20, shuffle=True, random_state=0)
        elif experiment == "yield_failure":
            cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=0)
        name = "kfold"
    return name, cv


if __name__ == "__main__":
    pass
