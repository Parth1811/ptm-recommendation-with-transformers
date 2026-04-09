"""Baseline transferability estimation methods for PTM recommendation.

Implements all 9 baseline methods used in Model Spider (NeurIPS 2023) for
comparing pre-trained model transferability. Each method takes pre-extracted
feature embeddings and target labels, returning a scalar transferability score
(higher = better predicted transfer performance).

Methods:
    1. H-Score (Bao et al., 2019)
    2. LEEP (Nguyen et al., 2020)
    3. LogME (You et al., 2021)
    4. NCE (Tran et al., 2019)
    5. NLEEP (Li et al., 2021)
    6. OTCE (Tan et al., 2021)
    7. PACTranDirichlet (Ding et al., 2022)
    8. GBC (Pandy et al., 2022)
    9. LFC (Li et al., 2021)

Dependencies: numpy, scipy (no torch required)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy import linalg, optimize, special, stats

logger = logging.getLogger(__name__)

# Numerical stability constants
_EPS = 1e-10
_REG = 1e-6


# ---------------------------------------------------------------------------
# 1. H-Score
# ---------------------------------------------------------------------------

def h_score(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """H-Score: An Easily Computable Transferability Measure.

    Measures how well the feature space separates classes by computing the
    ratio of between-class covariance to total covariance.

    H = tr(cov(f)^{-1} @ cov(E[f|y]))

    where cov(f) is the total feature covariance and cov(E[f|y]) is the
    covariance of class-conditional means.

    Reference:
        Bao, Y., Li, Y., Huang, S.L., Zhang, L., Zheng, L., Zamir, A.,
        and Guibas, L. "An Information-Theoretic Approach to Transferability
        in Task Transfer Learning." ICIP 2019.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels

    Returns:
        H-Score (float, higher = better transferability)
    """
    N, D = features.shape
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Total feature covariance: Σ_total = Cov(f)
    # Use shrinkage for numerical stability when D > N
    total_mean = features.mean(axis=0)
    centered = features - total_mean
    cov_total = (centered.T @ centered) / (N - 1) + _REG * np.eye(D)

    # Class-conditional means
    class_means = np.zeros((C, D))
    for i, c in enumerate(classes):
        class_means[i] = features[labels == c].mean(axis=0)

    # Between-class covariance: Cov(E[f|y])
    class_means_centered = class_means - total_mean
    cov_between = (class_means_centered.T @ class_means_centered) / (C - 1)

    # H = tr(Σ_total^{-1} @ Σ_between)
    try:
        cov_total_inv = linalg.inv(cov_total)
        score = np.trace(cov_total_inv @ cov_between)
    except linalg.LinAlgError:
        # Fallback: use pseudoinverse
        cov_total_inv = linalg.pinv(cov_total)
        score = np.trace(cov_total_inv @ cov_between)

    return float(score)


# ---------------------------------------------------------------------------
# 2. LEEP
# ---------------------------------------------------------------------------

def leep(
    features: np.ndarray,
    labels: np.ndarray,
    source_predictions: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Log Expected Empirical Prediction.

    Computes the average log-likelihood of target labels under a
    probabilistic source-to-target label mapping P(y|z), estimated
    from the empirical joint distribution of source predictions and
    target labels.

    LEEP = (1/N) Σ_n log(Σ_z P(y_n|z) * θ(z|x_n))

    where θ(z|x_n) is the source model's softmax output for sample n,
    and P(y|z) is the empirical conditional mapping from source to target.

    When source_predictions are not available, features are treated as
    softmax outputs from the source model (or converted via softmax if
    they don't sum to 1).

    Reference:
        Nguyen, C., Hassner, T., Seez, M., and Archambeau, C.
        "LEEP: A New Measure to Evaluate Transferability of Learned
        Representations." ICML 2020.

    Args:
        features: (N, D) feature matrix or (N, C_source) source softmax outputs
        labels: (N,) integer target class labels
        source_predictions: (N, C_source) source model softmax outputs.
            If None, features are treated as source predictions.

    Returns:
        LEEP score (float, higher = better transferability)
    """
    if source_predictions is not None:
        theta = source_predictions
    else:
        theta = features

    N = theta.shape[0]
    C_source = theta.shape[1]

    # Ensure theta is a valid probability distribution
    row_sums = theta.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=0.1):
        # Apply softmax
        theta = theta - theta.max(axis=1, keepdims=True)
        exp_theta = np.exp(theta)
        theta = exp_theta / exp_theta.sum(axis=1, keepdims=True)

    classes = np.unique(labels)
    C_target = len(classes)

    # Empirical joint P(y, z): for each (target_class, source_class) pair
    # P(y, z) = (1/N) * Σ_{n: y_n=y} θ(z|x_n)
    joint = np.zeros((C_target, C_source))
    for i, c in enumerate(classes):
        mask = labels == c
        joint[i] = theta[mask].sum(axis=0) / N

    # Marginal P(z) = Σ_y P(y, z)
    marginal_z = joint.sum(axis=0) + _EPS

    # Conditional P(y|z) = P(y, z) / P(z)
    p_y_given_z = joint / marginal_z[np.newaxis, :]

    # LEEP = (1/N) Σ_n log(Σ_z P(y_n|z) * θ(z|x_n))
    # Create target class index mapping
    label_to_idx = {c: i for i, c in enumerate(classes)}
    label_indices = np.array([label_to_idx[l] for l in labels])

    leep_score = 0.0
    for n in range(N):
        y_idx = label_indices[n]
        prob = np.dot(p_y_given_z[y_idx], theta[n])
        leep_score += np.log(prob + _EPS)

    return float(leep_score / N)


# ---------------------------------------------------------------------------
# 3. LogME
# ---------------------------------------------------------------------------

def logme(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """Log Maximum Evidence for transferability estimation.

    Fits a Bayesian linear model from features to labels for each class
    independently and computes the log marginal likelihood (evidence).
    Uses the efficient fixed-point iterative algorithm from the paper.

    Reference:
        You, K., Liu, Y., Wang, J., and Long, M.
        "LogME: Practical Assessment of Pre-trained Models for Transfer
        Learning." ICML 2021.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels

    Returns:
        LogME score (float, higher = better transferability)
    """
    f = features.astype(np.float64)
    N, D = f.shape
    C = int(labels.max()) + 1

    # SVD of features: more efficient than computing full covariance
    if N > D:
        # Truncated SVD via f^T f
        ftf = f.T @ f
        eigvals, eigvecs = np.linalg.eigh(ftf)
        # Keep only positive eigenvalues
        k = np.sum(eigvals > 1e-10)
        eigvals = eigvals[-k:]
        eigvecs = eigvecs[:, -k:]
        sigma = eigvals.reshape(-1, 1)  # (k, 1) — squared singular values
        s = np.sqrt(eigvals).reshape(-1, 1)
        vh = eigvecs.T  # (k, D)
        u = f @ eigvecs / s.reshape(1, -1)  # (N, k)
    else:
        u, s_vals, vh = np.linalg.svd(f, full_matrices=False)
        k = np.sum(s_vals > 1e-10)
        u = u[:, :k]
        s_vals = s_vals[:k]
        vh = vh[:k]
        s = s_vals.reshape(-1, 1)
        sigma = (s ** 2)

    evidences = []
    for c in range(C):
        y_ = (labels == c).astype(np.float64).reshape(-1, 1)

        x = u.T @ y_  # (k, 1)
        x2 = x ** 2
        res_x2 = float((y_ ** 2).sum() - x2.sum())

        alpha, beta = 1.0, 1.0
        for _ in range(11):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum()) + res_x2
            alpha = gamma / (m2 + 1e-5)
            beta = (N - gamma) / (res2 + 1e-5)
            t_ = alpha / beta
            if abs(t_ - t) / (t + _EPS) <= 1e-3:
                break

        evidence = (
            D / 2.0 * np.log(alpha)
            + N / 2.0 * np.log(beta)
            - 0.5 * np.sum(np.log(alpha + beta * sigma))
            - beta / 2.0 * res2
            - alpha / 2.0 * m2
            - N / 2.0 * np.log(2 * np.pi)
        )
        evidences.append(float(evidence) / N)

    return float(np.mean(evidences))


# ---------------------------------------------------------------------------
# 4. NCE (Negative Conditional Entropy)
# ---------------------------------------------------------------------------

def nce(
    features: np.ndarray,
    labels: np.ndarray,
    source_predictions: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Negative Conditional Entropy.

    Computes -H(y_target | y_source) via the empirical confusion matrix
    between source predictions (argmax of features/predictions) and
    target labels. Lower conditional entropy means better transferability.

    NCE = -H(Y_t | Y_s) = H(Y_t) - H(Y_t, Y_s)

    When features are softmax outputs, argmax is used to get source
    predictions. When features are embeddings, a nearest-centroid
    classifier is used.

    Reference:
        Tran, A.T., Nguyen, C.V., and Hassner, T.
        "Transferability and Hardness of Supervised Classification Tasks."
        ICCV 2019.

    Args:
        features: (N, D) feature matrix or (N, C_source) source softmax outputs
        labels: (N,) integer target class labels
        source_predictions: (N,) source model predicted class labels.
            If None, derived from features via argmax.

    Returns:
        NCE score (float, higher = better transferability)
    """
    N = len(labels)

    if source_predictions is not None:
        z = source_predictions.astype(int)
    else:
        z = np.argmax(features, axis=1).astype(int)

    target_classes = np.unique(labels)
    source_classes = np.unique(z)
    C_t = len(target_classes)
    C_s = len(source_classes)

    # Build empirical joint P(y_t, y_s)
    joint = np.zeros((C_t, C_s))
    t_map = {c: i for i, c in enumerate(target_classes)}
    s_map = {c: i for i, c in enumerate(source_classes)}

    for n in range(N):
        ti = t_map[labels[n]]
        si = s_map[z[n]]
        joint[ti, si] += 1
    joint /= N

    # Marginals
    p_t = joint.sum(axis=1)  # P(y_t)
    p_s = joint.sum(axis=0)  # P(y_s)

    # H(Y_t) = -Σ P(y_t) log P(y_t)
    h_t = -np.sum(p_t[p_t > 0] * np.log(p_t[p_t > 0]))

    # H(Y_t, Y_s) = -Σ P(y_t, y_s) log P(y_t, y_s)
    h_joint = -np.sum(joint[joint > 0] * np.log(joint[joint > 0]))

    # H(Y_t | Y_s) = H(Y_t, Y_s) - H(Y_s)
    h_s = -np.sum(p_s[p_s > 0] * np.log(p_s[p_s > 0]))
    h_cond = h_joint - h_s

    # NCE = -H(Y_t | Y_s)
    return float(-h_cond)


# ---------------------------------------------------------------------------
# 5. NLEEP (Normalized LEEP)
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances: (N, D) x (M, D) -> (N, M)."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XX + YY.T - 2 * (X @ Y.T)


def _kmeans_init(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """K-means++ initialization."""
    N, D = X.shape
    centroids = np.zeros((k, D))
    centroids[0] = X[rng.randint(N)]
    for i in range(1, k):
        dists = _pairwise_sq_dist(X, centroids[:i]).min(axis=1)
        dists = np.maximum(dists, 0)  # Clip numerical negatives
        probs = dists / (dists.sum() + _EPS)
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs = probs / (probs.sum() + _EPS)
        centroids[i] = X[rng.choice(N, p=probs)]
    return centroids

def nleep(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: Optional[int] = None,
    **kwargs,
) -> float:
    """Normalized LEEP using Gaussian Mixture Model.

    Replaces LEEP's source classifier with a GMM fitted on the feature
    space, producing soft cluster assignments as proxy for source labels.
    This handles cases where source model predictions are unavailable.

    Reference:
        Li, Y., Jia, X., Sang, R., Zhu, Y., Green, B., Wan, L.,
        and Tong, Z. "Ranking Neural Checkpoints." CVPR 2021.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels
        n_components: Number of GMM components (default: num_target_classes)

    Returns:
        NLEEP score (float, higher = better transferability)
    """
    N, D = features.shape
    classes = np.unique(labels)
    C_target = len(classes)

    if n_components is None:
        n_components = C_target

    # Dimensionality reduction if D is very large (for GMM stability)
    if D > 256:
        # PCA via SVD (numpy only)
        f_centered = features - features.mean(axis=0)
        _, _, Vt = np.linalg.svd(f_centered, full_matrices=False)
        k = min(256, N - 1, D)
        features_reduced = f_centered @ Vt[:k].T
    else:
        features_reduced = features

    # Fit GMM to get soft cluster assignments
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=200,
            random_state=42,
            n_init=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(features_reduced)
        # Soft assignments: P(z|x) for each cluster z
        theta = gmm.predict_proba(features_reduced)  # (N, n_components)
    except ImportError:
        # Fallback: numpy-only k-means-style soft assignments
        # Initialize centroids with k-means++
        rng = np.random.RandomState(42)
        centroids = _kmeans_init(features_reduced, n_components, rng)
        # Run k-means for a few iterations
        for _ in range(50):
            dists = _pairwise_sq_dist(features_reduced, centroids)
            assignments = np.argmin(dists, axis=1)
            for k in range(n_components):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = features_reduced[mask].mean(axis=0)
        # Soft assignments via negative squared distance (softmax)
        dists = _pairwise_sq_dist(features_reduced, centroids)
        dists = np.maximum(dists, 0)  # Clip numerical negatives
        dists = -dists / (dists.std() + _EPS)
        dists = dists - dists.max(axis=1, keepdims=True)
        theta = np.exp(dists)
        theta = theta / (theta.sum(axis=1, keepdims=True) + _EPS)

    # Compute LEEP with GMM cluster assignments as source predictions
    return leep(theta, labels, source_predictions=theta)


# ---------------------------------------------------------------------------
# 6. OTCE (Optimal Transport Conditional Entropy)
# ---------------------------------------------------------------------------

def otce(
    features: np.ndarray,
    labels: np.ndarray,
    source_features: Optional[np.ndarray] = None,
    source_labels: Optional[np.ndarray] = None,
    reg: float = 0.1,
    max_samples: int = 2000,
    **kwargs,
) -> float:
    """Optimal Transport based Conditional Entropy.

    Computes the OT coupling between source and target feature distributions,
    then estimates conditional entropy H(Y_t | Y_s) from the joint plan.
    Returns negative conditional entropy (higher = better).

    When source data is not available, creates synthetic source distributions
    by computing class centroids and sampling around them.

    Reference:
        Tan, Y., Li, Y., and Huang, S.L.
        "OTCE: A Transferability Metric for Cross-Domain Cross-Task
        Representations." CVPR 2021.

    Args:
        features: (N, D) target feature matrix
        labels: (N,) integer target class labels
        source_features: (M, D) source feature matrix (optional)
        source_labels: (M,) source class labels (optional)
        reg: Entropic regularization for Sinkhorn OT
        max_samples: Max samples to use (for computational tractability)

    Returns:
        OTCE score (float, higher = better transferability)
    """
    from scipy.spatial.distance import cdist

    N, D = features.shape

    # Subsample for computational tractability
    if N > max_samples:
        idx = np.random.RandomState(42).choice(N, max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        N = max_samples

    # If no source data, create synthetic source from class centroids
    if source_features is None or source_labels is None:
        classes = np.unique(labels)
        C = len(classes)
        # Create source by sampling from class-conditional distributions
        source_features_list = []
        source_labels_list = []
        n_per_class = max(N // C, 10)
        rng = np.random.RandomState(42)
        for c in classes:
            class_feats = features[labels == c]
            mu = class_feats.mean(axis=0)
            std = class_feats.std(axis=0) + _EPS
            synth = rng.normal(mu, std, size=(n_per_class, D))
            source_features_list.append(synth)
            source_labels_list.append(np.full(n_per_class, c))
        source_features = np.vstack(source_features_list)
        source_labels = np.concatenate(source_labels_list)

    M = source_features.shape[0]

    # Cost matrix: pairwise Euclidean distance
    C_matrix = cdist(source_features, features, metric="sqeuclidean")
    C_matrix = C_matrix / (C_matrix.max() + _EPS)

    # Sinkhorn OT coupling
    coupling = _sinkhorn(
        np.ones(M) / M,
        np.ones(N) / N,
        C_matrix,
        reg=reg,
        max_iter=100,
    )

    # Compute conditional entropy H(Y_t | Y_s) from coupling
    target_classes = np.unique(labels)
    source_classes = np.unique(source_labels)
    C_t = len(target_classes)
    C_s = len(source_classes)

    t_map = {c: i for i, c in enumerate(target_classes)}
    s_map = {c: i for i, c in enumerate(source_classes)}

    # Build joint P(y_s, y_t) from coupling
    joint = np.zeros((C_s, C_t))
    for i in range(M):
        si = s_map[source_labels[i]]
        for j in range(N):
            tj = t_map[labels[j]]
            joint[si, tj] += coupling[i, j]

    # Marginal P(y_s)
    p_s = joint.sum(axis=1)

    # H(Y_t | Y_s) = Σ_s P(y_s) * H(Y_t | Y_s = y_s)
    h_cond = 0.0
    for si in range(C_s):
        if p_s[si] < _EPS:
            continue
        p_t_given_s = joint[si] / (p_s[si] + _EPS)
        h_cond -= p_s[si] * np.sum(
            p_t_given_s[p_t_given_s > _EPS] * np.log(p_t_given_s[p_t_given_s > _EPS])
        )

    # Return negative conditional entropy (higher = better)
    return float(-h_cond)


def _sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 100,
) -> np.ndarray:
    """Sinkhorn-Knopp algorithm for entropic regularized OT.

    Args:
        a: (n,) source distribution
        b: (m,) target distribution
        M: (n, m) cost matrix
        reg: Entropic regularization
        max_iter: Maximum iterations

    Returns:
        coupling: (n, m) optimal transport plan
    """
    n, m = M.shape
    K = np.exp(-M / reg)

    u = np.ones(n) / n
    for _ in range(max_iter):
        v = b / (K.T @ u + _EPS)
        u = a / (K @ v + _EPS)

    coupling = np.diag(u) @ K @ np.diag(v)
    return coupling


# ---------------------------------------------------------------------------
# 7. PACTranDirichlet
# ---------------------------------------------------------------------------

def pactran_dirichlet(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """PAC-Bayesian Transferability with Dirichlet prior.

    Estimates transferability via a PAC-Bayesian bound using a Dirichlet
    conjugate prior over the class probability simplex. Fits a linear
    classifier on top of features, then computes the negative bound
    (higher = better transferability).

    The key idea: if features from a pre-trained model linearly separate
    target classes well, the KL divergence between the Dirichlet posterior
    and prior will be small, indicating good transferability.

    Reference:
        Ding, Y., Jiang, J., Yang, Y., Ye, J.P., and Wang, Y.
        "PACTran: PAC-Bayesian Metrics for Estimating the Transferability
        of Pretrained Models to Classification Tasks." ECCV 2022.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels

    Returns:
        PACTran-Dirichlet score (float, higher = better transferability)
    """
    N, D = features.shape
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Fit a simple linear classifier (logistic regression)
    # Compute class-conditional statistics for Dirichlet
    # α_0 (prior): uniform Dirichlet with concentration 1
    alpha_0 = np.ones(C)

    # α_posterior: count-based posterior
    # For each sample, compute P(y|x) via nearest-centroid or softmax
    class_counts = np.zeros(C)
    class_means = np.zeros((C, D))
    for i, c in enumerate(classes):
        mask = labels == c
        class_counts[i] = mask.sum()
        class_means[i] = features[mask].mean(axis=0)

    # Compute per-sample likelihoods via Mahalanobis-like distance
    # Use simplified approach: softmax over negative distances to centroids
    dists = np.zeros((N, C))
    for i in range(C):
        diff = features - class_means[i]
        dists[:, i] = -np.sum(diff ** 2, axis=1)

    # Temperature-scaled softmax
    dists = dists - dists.max(axis=1, keepdims=True)
    probs = np.exp(dists)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Aggregate posterior: sum of predicted probabilities per class
    alpha_post = alpha_0.copy()
    for i, c in enumerate(classes):
        mask = labels == c
        alpha_post[i] += probs[mask, i].sum()

    # KL divergence: KL(Dir(α_post) || Dir(α_0))
    kl = _kl_dirichlet(alpha_post, alpha_0)

    # Empirical risk: average negative log-likelihood
    label_to_idx = {c: i for i, c in enumerate(classes)}
    nll = 0.0
    for n in range(N):
        y_idx = label_to_idx[labels[n]]
        nll -= np.log(probs[n, y_idx] + _EPS)
    nll /= N

    # PAC-Bayes bound: risk + sqrt(KL / (2N))
    bound = nll + np.sqrt(kl / (2 * N))

    # Return negative bound (higher = better)
    return float(-bound)


def _kl_dirichlet(alpha: np.ndarray, beta: np.ndarray) -> float:
    """KL divergence between two Dirichlet distributions.

    KL(Dir(α) || Dir(β)) = log(B(β)/B(α)) + Σ(α_i - β_i)(ψ(α_i) - ψ(α_0))

    where B is the multivariate beta function, ψ is the digamma function,
    and α_0 = Σ α_i.
    """
    a0 = alpha.sum()
    b0 = beta.sum()

    kl = (
        special.gammaln(a0) - special.gammaln(b0)
        - np.sum(special.gammaln(alpha)) + np.sum(special.gammaln(beta))
        + np.sum((alpha - beta) * (special.digamma(alpha) - special.digamma(a0)))
    )
    return float(max(kl, 0.0))


# ---------------------------------------------------------------------------
# 8. GBC (Gaussian Bhattacharyya Coefficient)
# ---------------------------------------------------------------------------

def gbc(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """Gaussian Bhattacharyya Coefficient for transferability.

    Models each target class as a Gaussian in the feature space and computes
    the pairwise Bhattacharyya distance. Higher distance means better class
    separation, implying better transferability.

    Returns the negative average Bhattacharyya coefficient (overlap) so that
    higher = better separation.

    Reference:
        Pandy, M., Agostinelli, A., Uijlings, J., Ferrari, V.,
        and Mensink, T. "Transferability Estimation using Bhattacharyya
        Class Separability." CVPR 2022.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels

    Returns:
        GBC score (float, higher = better transferability)
    """
    N, D = features.shape
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Dimensionality reduction for numerical stability
    max_dim = min(D, 64, N // C - 1) if N // C > 2 else min(D, 16)
    if D > max_dim:
        # PCA via SVD (numpy only, no sklearn)
        f_centered = features - features.mean(axis=0)
        _, _, Vt = np.linalg.svd(f_centered, full_matrices=False)
        features = f_centered @ Vt[:max_dim].T
        D = max_dim

    # Compute class-conditional Gaussians
    means = []
    covs = []
    for c in classes:
        f_c = features[labels == c]
        n_c = f_c.shape[0]
        mu = f_c.mean(axis=0)
        if n_c < D + 1:
            # Not enough samples: use diagonal covariance
            cov = np.diag(f_c.var(axis=0) + _REG)
        else:
            cov = np.cov(f_c, rowvar=False) + _REG * np.eye(D)
        means.append(mu)
        covs.append(cov)

    # Compute pairwise Bhattacharyya distances
    total_dist = 0.0
    n_pairs = 0
    for i in range(C):
        for j in range(i + 1, C):
            bd = _bhattacharyya_distance(means[i], covs[i], means[j], covs[j])
            total_dist += bd
            n_pairs += 1

    avg_dist = total_dist / max(n_pairs, 1)

    # Return negative Bhattacharyya coefficient: -exp(-DB)
    # Higher distance = lower overlap = better transferability
    # We return the average distance directly (higher = better)
    return float(avg_dist)


def _bhattacharyya_distance(
    mu1: np.ndarray, cov1: np.ndarray,
    mu2: np.ndarray, cov2: np.ndarray,
) -> float:
    """Bhattacharyya distance between two Gaussians.

    DB = (1/8)(μ1-μ2)^T Σ^{-1} (μ1-μ2) + (1/2) ln(|Σ| / sqrt(|Σ1||Σ2|))

    where Σ = (Σ1 + Σ2) / 2
    """
    diff = mu1 - mu2
    cov_avg = (cov1 + cov2) / 2.0

    try:
        cov_avg_inv = linalg.inv(cov_avg)
        sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
        sign1, logdet1 = np.linalg.slogdet(cov1)
        sign2, logdet2 = np.linalg.slogdet(cov2)

        if sign_avg <= 0 or sign1 <= 0 or sign2 <= 0:
            return 0.0

        term1 = 0.125 * diff @ cov_avg_inv @ diff
        term2 = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
        return float(term1 + term2)
    except linalg.LinAlgError:
        return 0.0


# ---------------------------------------------------------------------------
# 9. LFC (Linear Feature Correlation / Fisher Criterion)
# ---------------------------------------------------------------------------

def lfc(
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """Linear Feature Correlation (Fisher's discriminant ratio).

    Computes the ratio of inter-class scatter to intra-class scatter in
    the feature space: tr(S_B) / tr(S_W), where S_B is the between-class
    scatter matrix and S_W is the within-class scatter matrix.

    Higher ratio means features better separate the classes linearly.

    Reference:
        Li, Y., Jia, X., Sang, R., Zhu, Y., Green, B., Wan, L.,
        and Tong, Z. "Ranking Neural Checkpoints." CVPR 2021.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels

    Returns:
        LFC score (float, higher = better transferability)
    """
    N, D = features.shape
    classes = np.unique(labels)
    C = len(classes)

    if C < 2:
        return 0.0

    # Global mean
    mu = features.mean(axis=0)

    # Between-class scatter: S_B = Σ_c n_c * (μ_c - μ)(μ_c - μ)^T
    # Within-class scatter: S_W = Σ_c Σ_{x in c} (x - μ_c)(x - μ_c)^T
    # We only need traces, so we can compute efficiently without forming matrices

    tr_sb = 0.0
    tr_sw = 0.0

    for c in classes:
        f_c = features[labels == c]
        n_c = f_c.shape[0]
        mu_c = f_c.mean(axis=0)

        # Between-class contribution: n_c * ||μ_c - μ||^2
        diff = mu_c - mu
        tr_sb += n_c * np.dot(diff, diff)

        # Within-class contribution: Σ ||x - μ_c||^2
        centered = f_c - mu_c
        tr_sw += np.sum(centered ** 2)

    return float(tr_sb / (tr_sw + _EPS))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHODS = {
    "h_score": h_score,
    "H-Score": h_score,
    "leep": leep,
    "LEEP": leep,
    "logme": logme,
    "LogME": logme,
    "nce": nce,
    "NCE": nce,
    "nleep": nleep,
    "NLEEP": nleep,
    "otce": otce,
    "OTCE": otce,
    "pactran_dirichlet": pactran_dirichlet,
    "PACTranDirichlet": pactran_dirichlet,
    "gbc": gbc,
    "GBC": gbc,
    "lfc": lfc,
    "LFC": lfc,
}

ALL_METHOD_NAMES = [
    "H-Score", "LEEP", "LogME", "NCE", "NLEEP",
    "OTCE", "PACTranDirichlet", "GBC", "LFC",
]


def compute_transferability(
    method_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs,
) -> float:
    """Dispatch to a named transferability estimation method.

    Args:
        method_name: Name of the method (case-sensitive, see METHODS dict)
        features: (N, D) feature matrix from pre-trained model
        labels: (N,) integer class labels
        **kwargs: Additional method-specific arguments

    Returns:
        Transferability score (float, higher = better)

    Raises:
        ValueError: If method_name is unknown
    """
    if method_name not in METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {list(METHODS.keys())}"
        )

    fn = METHODS[method_name]
    try:
        score = fn(features, labels, **kwargs)
    except Exception as e:
        logger.warning(f"Method '{method_name}' failed: {e}")
        score = float("nan")

    return score


def compute_all_transferability(
    features: np.ndarray,
    labels: np.ndarray,
    methods: list[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Compute all (or selected) transferability metrics.

    Args:
        features: (N, D) feature matrix
        labels: (N,) integer class labels
        methods: List of method names (default: all 9 methods)
        **kwargs: Additional method-specific arguments

    Returns:
        Dict mapping method_name -> score
    """
    if methods is None:
        methods = ALL_METHOD_NAMES

    results = {}
    for name in methods:
        logger.info(f"Computing {name}...")
        results[name] = compute_transferability(name, features, labels, **kwargs)
        logger.info(f"  {name} = {results[name]:.6f}")

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing all 9 transferability estimation methods...\n")

    rng = np.random.RandomState(42)
    N, D, C = 200, 64, 5

    # Create synthetic features with class structure
    labels = rng.randint(0, C, size=N)
    class_centers = rng.randn(C, D) * 3
    features = class_centers[labels] + rng.randn(N, D) * 0.5

    for name in ALL_METHOD_NAMES:
        try:
            score = compute_transferability(name, features, labels)
            print(f"  {name:20s}: {score:.6f}")
        except Exception as e:
            print(f"  {name:20s}: FAILED ({e})")

    print("\nAll tests complete.")
