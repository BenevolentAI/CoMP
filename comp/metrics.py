import numpy as np
from scipy.stats import chisquare
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import NearestNeighbors


def compute_mmd(x1, x2, kernel=None):
    """
    Maximum mean discrepancy (MMD)

    Args:
        x1 (np.ndarray): n x m array representing the first sample
        x2 (np.ndarray): n x m array representing the second sample
        kernel: the kernel function. If not provided, this will use a RBF kernel with length_scale=1

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    kernel = kernel or RBF(length_scale=1.0)
    x1x1 = kernel(x1, x1)
    x1x2 = kernel(x1, x2)
    x2x2 = kernel(x2, x2)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
    return diff


def knn_metric(
    features, queries, labels, class_partition, n_neighbours=50, return_counts=False
):
    """
    Compute the nearest neighbour metric for datasets composed of samples that are in one of
    two classes and have one of a number of labels. For example with the Celligner dataset
    contains tumor and cell-line samples that are each labelled with a disease.

    The metric counts the porportion of each query sample's nearest neighbours that are both:
        a) from the other clas of samples
        b) labelled with the same disease

    The counts of both conditions and the count of their conjunction is also optionally returned
    if `return_counts` is `True`. This is returned as a (len(queries) x 3) array, where the
    columns are the counts of the conditions [a, b, (a & b)] for each query element.

    Args:
        features: a data frame containing the representations of the entities
        queries: an array indicating which of the entities in features to find and score the
                 nearest neighbours for.
        labels: An array of labels for each entity in features; The metric counts
                        neighbours of each query entity that have the same label.
        class_partition: A boolean array splitting the entities in features into 2 classes;
                         The metric counts neighbours of each query entity that are from
                         the other class according to this partition.
        n_neighbours: how many neighbours to consider
        return_counts: Also return an array of counts if true
    """
    knn = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(features)
    _, knnidx = knn.kneighbors(features[queries])
    # First column of nearest neighbours are the points themselves
    knn_self = knnidx[:, 0]
    knnidx = knnidx[:, 1:]

    # Count number of NNs that are a) from the other class, b) from the same disease type
    # counts: n_samples x {other_class_and_same_disease, other_class, same_disease}
    counts = np.zeros((knnidx.shape[0], 3), dtype=np.int32)
    for i in range(knnidx.shape[0]):
        idx = knn_self[i]
        for j in range(knnidx.shape[1]):
            class_diff = class_partition[idx] != class_partition[knnidx[i, j]]
            disease_match = labels[idx] == labels[knnidx[i, j]]
            counts[i, 0] += np.int32(class_diff and disease_match)
            counts[i, 1] += np.int32(class_diff)
            counts[i, 2] += np.int32(disease_match)

    knn_score = counts[:, 0].ravel() / n_neighbours
    if return_counts:
        return knn_score, counts
    else:
        return knn_score


def silhouette_coeff(features, queries, labels, class_partition, n_neighbours=50):
    """The silhouette coefficient

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6152897/#S10

    """
    knn = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(features)
    distances, knnidx = knn.kneighbors(features[queries])
    # First column of nearest neighbours are the points themselves
    # knn_self = knnidx[:, 0]
    # knnidx = knnidx[:, 1:]

    silhouette = np.zeros((knnidx.shape[0]))
    disease_silhouette = np.zeros((knnidx.shape[0]))
    for i in range(knnidx.shape[0]):
        idx = knnidx[i, 0]
        idx_neigh = knnidx[i, 1:]

        # No diseases
        neigh_partition = class_partition[idx_neigh]
        same_class = neigh_partition == class_partition[idx]
        other_class = np.invert(same_class)

        if sum(neigh_partition) == len(neigh_partition):
            silhouette[i] = 1
            continue
        elif sum(neigh_partition) == 0:
            silhouette[i] = -1
            continue

        ave_same_dist = np.mean(distances[i, 1:][same_class])
        ave_other_dist = np.mean(distances[i, 1:][other_class])
        if ave_same_dist < ave_other_dist:
            silhouette[i] = 1 - ave_same_dist / ave_other_dist
        elif ave_same_dist > ave_other_dist:
            silhouette[i] = ave_other_dist / ave_same_dist - 1
        else:
            silhouette[i] = 0

        # With diseases
        same_disease = labels[idx_neigh] == labels[idx]
        same_class_same_disease = same_class * same_disease
        other_class_same_disease = other_class * same_disease

        if sum(same_class_same_disease) == 0 and sum(other_class_same_disease) == 0:
            disease_silhouette[i] = 0
            continue
        elif sum(same_class_same_disease) == 0:
            disease_silhouette[i] = -1
            continue
        elif sum(other_class_same_disease) == 0:
            disease_silhouette[i] = 1
            continue

        ave_same_dist2 = np.mean(distances[i, 1:][same_class_same_disease])
        ave_other_dist2 = np.mean(distances[i, 1:][other_class_same_disease])
        if ave_same_dist2 < ave_other_dist2:
            disease_silhouette[i] = 1 - ave_same_dist2 / ave_other_dist2
        elif ave_same_dist2 > ave_other_dist2:
            disease_silhouette[i] = ave_other_dist2 / ave_same_dist2 - 1
        else:
            disease_silhouette[i] = 0

    return silhouette, disease_silhouette


def kbet(counts, expected_freq, n_neighbours, significance=0.05):
    """kBET metric
    0 = well mixed, 1 = not well mixed

    https://www.nature.com/articles/s41592-018-0254-1

    Args:
        counts (ndarray): A (n_queries,)-shaped array with counts of the OPPOSITE (NB!) class of the n_queries
        expected_freq (ndarray): The expected number in the kNN neighborhood of the OPPOSITE (NB!)
            class from the query under the null hypothesis.
        n_neighbours (list or int): The number of neighbours for query.
        significance (float): The significance threshold

    Returns:
        (float): The proportion of null hypotheses rejected.
    """
    if isinstance(n_neighbours, int):
        n_neighbours = [n_neighbours] * counts.shape[0]
    elif isinstance(n_neighbours, list) and len(n_neighbours) != counts.shape[0]:
        raise ValueError(
            f"len(n_neighbours) ( {len(n_neighbours)}) not equal to number of counts ({counts.shape[0]})"
        )
    p_vals = np.ones(counts.shape[0])
    for i in range(counts.shape[0]):
        f_obs = [counts[i], n_neighbours[i] + 1 - counts[i]]
        f_exp = [expected_freq[i], n_neighbours[i] + 1 - expected_freq[i]]
        _, p_val = chisquare(
            f_obs=f_obs,
            f_exp=f_exp,
            axis=None,
        )
        p_vals[i] = p_val
    return sum(p_vals < significance) / counts.shape[0]
