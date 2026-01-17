# Copyright 2024-2026 Nicholas Gigliotti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are derived from scikit-learn (https://github.com/scikit-learn/scikit-learn),
# which is licensed under the BSD 3-Clause License:
# https://opensource.org/licenses/BSD-3-Clause
#
# Modifications to the original code were made by Nicholas Gigliotti and are
# licensed under the Apache 2.0 License.

import logging

import torch

logger = logging.getLogger(__name__)


class IncrementalPCA:
    """Incremental principal components analysis (IPCA).

    Adapted from scikit-learn's `IncrementalPCA` (BSD 3-Clause License).

    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    Depending on the size of the input data, this algorithm can be much more
    memory efficient than a PCA, and allows sparse input.

    This algorithm has constant memory complexity, on the order
    of ``batch_size * n_features``, enabling use of np.memmap files without
    loading the entire file into memory. For sparse matrices, the input
    is converted to dense in batches (in order to be able to subtract the
    mean) which avoids storing the entire dense matrix at any one time.

    The computational overhead of each SVD is
    ``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
    remain in memory at a time. There will be ``n_samples / batch_size`` SVD
    computations to get the principal components, versus 1 large SVD of
    complexity ``O(n_samples * n_features ** 2)`` for PCA.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If ``n_components`` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    whiten : bool, default=False
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    copy : bool, default=True
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. Equivalently, the right singular
        vectors of the centered input data, parallel to its eigenvectors.
        The components are sorted by decreasing ``explained_variance_``.

    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If all components are stored, the sum of explained variances is equal
        to 1.0.

    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, aggregate over calls to ``partial_fit``.

    var_ : ndarray of shape (n_features,)
        Per-feature empirical variance, aggregate over calls to
        ``partial_fit``.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf.

    n_components_ : int
        The estimated number of components. Relevant when
        ``n_components=None``.

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    batch_size_ : int
        Inferred batch size from ``batch_size``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    PCA : Principal component analysis (PCA).
    KernelPCA : Kernel Principal component analysis (KPCA).
    SparsePCA : Sparse Principal Components Analysis (SparsePCA).
    TruncatedSVD : Dimensionality reduction using truncated SVD.

    Notes
    -----
    Implements the incremental PCA model from:
    *D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, May 2008.*
    See https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

    This model is an extension of the Sequential Karhunen-Loeve Transform from:
    :doi:`A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
    its Application to Images, IEEE Transactions on Image Processing, Volume 9,
    Number 8, pp. 1371-1374, August 2000. <10.1109/83.855432>`

    We have specifically abstained from an optimization used by authors of both
    papers, a QR decomposition used in specific situations to reduce the
    algorithmic complexity of the SVD. The source for this technique is
    *Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
    section 5.4.4, pp 252-253.*. This technique has been omitted because it is
    advantageous only when decomposing a matrix with ``n_samples`` (rows)
    >= 5/3 * ``n_features`` (columns), and hurts the readability of the
    implemented algorithm. This would be a good opportunity for future
    optimization, if it is deemed necessary.

    References
    ----------
    D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77,
    Issue 1-3, pp. 125-141, May 2008.

    G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
    Section 5.4.4, pp. 252-253.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        whiten: bool = False,
        copy: bool = True,
        batch_size: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = torch.device(device)

    def __repr__(self) -> str:
        return f"IncrementalPCA(n_components={self.n_components}, device={self.device})"

    def to(self, device: str | torch.device) -> "IncrementalPCA":
        self.device = torch.device(device)
        fitted_attrs = [
            "n_samples_seen_",
            "components_",
            "mean_",
            "singular_values_",
            "explained_variance_",
            "explained_variance_ratio_",
            "var_",
            "noise_variance_",
        ]
        for attr in fitted_attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    @torch.no_grad()
    def get_covariance(self) -> torch.Tensor:
        """Compute data covariance with the generative model using PyTorch.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : Tensor of shape (n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * torch.sqrt(exp_var[:, None])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = torch.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            torch.tensor(0.0, device=self.device),
        )
        cov = (components_.T * exp_var_diff) @ components_
        _add_to_diagonal(cov, self.noise_variance_)
        return cov

    @torch.no_grad()
    def get_precision(self) -> torch.Tensor:
        """Compute data precision matrix with the generative model using PyTorch.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : Tensor of shape (n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # Handle corner cases first
        if self.n_components_ == 0:
            return torch.eye(n_features, device=self.components_.device) / self.noise_variance_

        if self.noise_variance_ == 0.0:
            return torch.inverse(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * torch.sqrt(exp_var[:, None])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = torch.where(
            exp_var > self.noise_variance_,
            exp_var_diff,
            torch.tensor(0.0, device=exp_var.device),
        )
        precision = components_ @ components_.T / self.noise_variance_
        _add_to_diagonal(precision, 1.0 / exp_var_diff)
        precision = components_.T @ torch.inverse(precision) @ components_
        precision /= -(self.noise_variance_**2)
        _add_to_diagonal(precision, 1.0 / self.noise_variance_)
        return precision

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {Tensor} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : Tensor of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        if not hasattr(self, "components_"):
            raise ValueError(
                "This IncrementalPCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        X_transformed = X @ self.components_.T
        # Apply the centering after the projection.
        X_transformed -= self.mean_.view(1, -1) @ self.components_.T
        if self.whiten:
            # For some solvers, on rank deficient data, some components can have a variance
            # arbitrarily close to zero, leading to non-finite results when whitening.
            # To avoid this problem we clip the variance below.
            scale = torch.sqrt(self.explained_variance_)
            min_scale = torch.finfo(scale.dtype).eps
            scale = torch.clamp(scale, min=min_scale)
            X_transformed /= scale
        return X_transformed

    @torch.no_grad()
    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of components.

        Returns
        -------
        X_original : Tensor of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        if self.whiten:
            scaled_components = torch.sqrt(self.explained_variance_[:, None]) * self.components_
            return X @ scaled_components + self.mean_
        else:
            return X @ self.components_ + self.mean_

    def fit_transform(self, X: torch.Tensor, y: None = None) -> torch.Tensor:
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : {Tensor} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : Tensor of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        return self.fit(X, y).transform(X)

    @property
    def _n_features_out(self) -> int:
        """Number of transformed output features."""
        return self.components_.shape[0]

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: None = None) -> "IncrementalPCA":
        """Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = 0.0
        self.var_ = 0.0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None

        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(
            n_samples, self.batch_size_, min_batch_size=self.n_components or 0
        ):
            X_batch = X[batch]
            self.partial_fit(X_batch)

        return self

    @torch.no_grad()
    def partial_fit(self, X: torch.Tensor, y: None = None) -> "IncrementalPCA":
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        first_pass = not hasattr(self, "components_")
        if X.device != self.device:
            logger.debug(f"Moving X to device '{self.device}'")
            X = X.to(self.device)

        n_samples, n_features = X.shape
        if first_pass:
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not self.n_components <= n_features:
            raise ValueError(
                f"n_components={self.n_components} invalid for n_features={n_features}, need "
                "more rows than columns for IncrementalPCA processing"
            )
        elif not self.n_components <= n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal to "
                f"the batch number of samples {n_samples}."
            )
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] != self.n_components_):
            raise ValueError(
                f"Number of input features has changed from {self.components_.shape[0]} "
                f"to {self.n_components_} between calls to partial_fit! Try "
                "setting n_components to a fixed value."
            )

        # This is the first partial_fit
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.mean_ = torch.zeros(n_features, device=self.device)
            self.var_ = torch.zeros(n_features, device=self.device)

        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count=torch.as_tensor(self.n_samples_seen_, device=self.device).repeat(
                X.shape[1]
            ),
        )
        n_total_samples = n_total_samples[0]
        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = torch.sqrt((self.n_samples_seen_ / n_total_samples) * n_samples) * (
                self.mean_ - col_batch_mean
            )
            X = torch.vstack(
                (
                    self.singular_values_.reshape((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        # we already checked `self.n_components <= n_samples` above
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=self.device)
        return self


def gen_batches(n: int, batch_size: int, *, min_batch_size: int = 0) -> slice:  # type: ignore
    """Generator to create slices containing `batch_size` elements from 0 to `n`.

    Adapted from scikit-learn's `gen_batches` (BSD 3-Clause License).

    The last slice may contain less than `batch_size` elements, when
    `batch_size` does not divide `n`.

    Parameters
    ----------
    n : int
        Size of the sequence.
    batch_size : int
        Number of elements in each batch.
    min_batch_size : int, default=0
        Minimum number of elements in each batch.

    Yields
    ------
    slice of `batch_size` elements

    See Also
    --------
    gen_even_slices: Generator to create n_packs slices going up to n.

    Examples
    --------
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    >>> list(gen_batches(7, 3, min_batch_size=0))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(7, 3, min_batch_size=2))
    [slice(0, 3, None), slice(3, 7, None)]
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


@torch.no_grad()
def svd_flip(
    u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign correction to ensure deterministic output from SVD.

    Adapted from scikit-learn's `svd_flip` (BSD 3-Clause License).

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : Tensor
        Parameters u and v are the output of `torch.linalg.svd`, with matching inner
        dimensions so one can compute `torch.matmul(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : Tensor
        Parameters u and v are the output of `torch.linalg.svd`, with matching inner
        dimensions so one can compute `torch.matmul(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : Tensor
        Tensor u with adjusted columns and the same dimensions as u.

    v_adjusted : Tensor
        Tensor v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = torch.argmax(torch.abs(u.T), dim=1)
        signs = torch.sign(u.T[torch.arange(u.T.shape[0]), max_abs_u_cols])
        u *= signs
        if v is not None:
            v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_v_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[torch.arange(v.shape[0]), max_abs_v_rows])
        if u is not None:
            u *= signs
        v *= signs[:, None]
    return u, v


@torch.no_grad()
def _add_to_diagonal(array: torch.Tensor, value: float) -> torch.Tensor:
    """Add a value to the diagonal elements of a 2D tensor in PyTorch.

    Adapted from scikit-learn's `_add_to_diagonal` (BSD 3-Clause License).

    Parameters
    ----------
    array : Tensor of shape (n, n)
        Input tensor.
    value : float
        Value to add to the diagonal elements.

    Returns
    -------
    array : Tensor of shape (n, n)
        Input tensor with value added to the diagonal elements.
    """
    indices = torch.arange(array.size(0), device=array.device)
    array[indices, indices] += value
    return array


@torch.no_grad()
def _incremental_mean_and_var(
    X: torch.Tensor,
    last_mean: torch.Tensor,
    last_variance: torch.Tensor,
    last_sample_count: torch.Tensor,
    sample_weight: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate mean update and a Youngs and Cramer variance update using PyTorch.

    Adapted from scikit-learn's `_incremental_mean_and_var` (BSD 3-Clause License).

    If sample_weight is given, the weighted mean and variance is computed.

    Update a given mean and (possibly) variance according to new data given
    in X. last_mean is always required to compute the new mean.
    If last_variance is None, no variance is computed and None return for
    updated_variance.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : Tensor of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : Tensor of shape (n_features,)

    last_variance : Tensor of shape (n_features,)

    last_sample_count : Tensor of shape (n_features,)
        The number of samples encountered until now if sample_weight is None.
        If sample_weight is not None, this is the sum of sample_weight
        encountered.

    sample_weight : Tensor of shape (n_samples,) or None
        Sample weights. If None, compute the unweighted mean/variance.

    Returns
    -------
    updated_mean : Tensor of shape (n_features,)

    updated_variance : Tensor of shape (n_features,)
        None if last_variance was None.

    updated_sample_count : Tensor of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    X_nan_mask = torch.isnan(X)
    if torch.any(X_nan_mask):
        sum_op = torch.nansum
    else:
        sum_op = torch.sum
    if sample_weight is not None:
        new_sum = torch.matmul(
            sample_weight,
            torch.where(X_nan_mask, torch.tensor(0.0, device=X.device), X),
        )
        new_sample_count = torch.sum(sample_weight[:, None] * (~X_nan_mask), axis=0)
    else:
        new_sum = sum_op(X, axis=0)
        n_samples = X.shape[0]
        new_sample_count = n_samples - torch.sum(X_nan_mask, axis=0)

    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T
        if sample_weight is not None:
            correction = torch.matmul(
                sample_weight,
                torch.where(X_nan_mask, torch.tensor(0.0, device=X.device), temp),
            )
            temp **= 2
            new_unnormalized_variance = torch.matmul(
                sample_weight,
                torch.where(X_nan_mask, torch.tensor(0.0, device=X.device), temp),
            )
        else:
            correction = sum_op(temp, axis=0)
            temp **= 2
            new_unnormalized_variance = sum_op(temp, axis=0)

        new_unnormalized_variance -= correction**2 / new_sample_count

        last_unnormalized_variance = last_variance * last_sample_count

        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance
            + new_unnormalized_variance
            + last_over_new_count
            / updated_sample_count
            * (last_sum / last_over_new_count - new_sum) ** 2
        )

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count
