import numpy as np
from pymatting_aot.aot import _make_tree, _find_knn


class KDTree(object):
    """KDTree implementation"""

    def __init__(self, data_points, min_leaf_size=8):
        """Constructs a KDTree for given data points. The implementation currently only supports data type `np.float32`.

        Parameters
        ----------
        data_points: numpy.ndarray (of type `np.float32`)
            Dataset with shape :math:`n \\times d`, where :math:`n` is the number of data points in the data set and :math:`d` is the dimension of each data point
        min_leaf_size: int
            Minimum number of nodes in a leaf, defaults to 8

        Example
        -------
        >>> from pymatting import *
        >>> import numpy as np
        >>> data_set = np.random.randn(100, 2)
        >>> tree = KDTree(data_set.astype(np.float32))
        """
        assert data_points.dtype == np.float32

        n_data, dimension = data_points.shape

        max_nodes = 2 * ((n_data + min_leaf_size - 1) // min_leaf_size)

        self.i0_inds = np.empty(max_nodes, np.int64)
        self.i1_inds = np.empty(max_nodes, np.int64)
        self.less_inds = np.empty(max_nodes, np.int64)
        self.more_inds = np.empty(max_nodes, np.int64)
        self.split_dims = np.empty(max_nodes, np.int64)
        self.bounds = np.empty((max_nodes, 2, dimension), np.float32)
        self.split_values = np.empty(max_nodes, np.float32)
        self.shuffled_data_points = data_points.copy()
        self.shuffled_indices = np.arange(n_data).astype(np.int64)

        self.n_nodes = _make_tree(
            self.i0_inds,
            self.i1_inds,
            self.less_inds,
            self.more_inds,
            self.split_dims,
            self.bounds,
            self.split_values,
            self.shuffled_data_points,
            self.shuffled_indices,
            min_leaf_size,
        )

    def query(self, query_points, k):
        """Query the tree

        Parameters
        ----------
        query_points: numpy.ndarray (of type `np.float32`)
            Data points for which the next neighbours should be calculated
        k: int
            Number of neighbors to find

        Returns
        -------
        distances: numpy.ndarray
            Distances to the neighbors
        indices: numpy.ndarray
            Indices of the k nearest neighbors in original data array

        Example
        -------
        >>> from pymatting import *
        >>> import numpy as np
        >>> data_set = np.random.randn(100, 2)
        >>> tree = KDTree(data_set.astype(np.float32))
        >>> tree.query(np.array([[0.5,0.5]], dtype=np.float32), k=3)
        (array([[0.14234178, 0.15879704, 0.26760164]], dtype=float32), array([[29, 21, 20]]))
        """
        assert query_points.dtype == np.float32

        n_query = query_points.shape[0]

        squared_distances = np.empty((n_query, k), np.float32)
        indices = np.empty((n_query, k), np.int64)

        _find_knn(
            self.i0_inds,
            self.i1_inds,
            self.less_inds,
            self.more_inds,
            self.split_dims,
            self.bounds,
            self.split_values,
            self.shuffled_data_points,
            query_points,
            indices,
            squared_distances,
            k,
        )

        indices = self.shuffled_indices[indices]
        distances = np.sqrt(squared_distances)

        return distances, indices


def knn(data_points, query_points, k):
    """Find k nearest neighbors in a data set. The implementation currently only supports data type `np.float32`.

    Parameters
    ----------
    data_points: numpy.ndarray (of type `np.float32`)
        Dataset with shape :math:`n \\times d`, where :math:`n` is the number of data points in the data set and :math:`d` is the dimension of each data point
    query_points: numpy.ndarray (of type `np.float32`)
        Data points for which the next neighbours should be calculated
    k: int
        Number of neighbors to find

    Returns
    -------
    distances: numpy.ndarray
        Distances to the neighbors
    indices: numpy.ndarray
        Indices of the k nearest neighbors in original data array

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> data_set = np.random.randn(100, 2)
    >>> knn(data_set.astype(np.float32), np.array([[0.5,0.5]], dtype=np.float32), k=2)
    (array([[0.16233477, 0.25393516]], dtype=float32), array([[25, 17]]))
    """
    tree = KDTree(data_points)
    return tree.query(query_points, k)
