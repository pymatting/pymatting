import numpy as np
from numba import njit, prange

# Numba support for object pointers is currently (Q4 2019) wonky,
# which is why plain arrays with indices are used instead.

@njit("i8(i8[:], i8[:], i8[:], i8[:], i8[:], f4[:, :, :], f4[:], f4[:, :], i8[:], i8)", cache=True, nogil=True)
def _make_tree(
    i0_inds,
    i1_inds,
    less_inds,
    more_inds,
    split_dims,
    bounds,
    split_values,
    points,
    indices,
    min_leaf_size,
):
    dimension = points.shape[1]
    # Expect log2(len(points) / min_leaf_size) depth, 1000 should be plenty
    stack = np.empty(1000, np.int64)
    stack_size = 0
    n_nodes = 0
    # min_leaf_size <= leaf_node_size < max_leaf_size
    max_leaf_size = 2 * min_leaf_size

    # Push i0, i1, i_node
    stack[stack_size] = 0
    stack_size += 1
    stack[stack_size] = points.shape[0]
    stack_size += 1
    stack[stack_size] = n_nodes
    n_nodes += 1
    stack_size += 1

    # While there are more tree nodes to process recursively
    while stack_size > 0:
        stack_size -= 1
        i_node = stack[stack_size]
        stack_size -= 1
        i1 = stack[stack_size]
        stack_size -= 1
        i0 = stack[stack_size]

        lo = bounds[i_node, 0]
        hi = bounds[i_node, 1]

        for d in range(dimension):
            lo[d] = points[i0, d]
            hi[d] = points[i0, d]

        # Find lower and upper bounds of points for each dimension
        for i in range(i0 + 1, i1):
            for d in range(dimension):
                lo[d] = min(lo[d], points[i, d])
                hi[d] = max(hi[d], points[i, d])

        # Done if node is small
        if i1 - i0 <= max_leaf_size:
            i0_inds[i_node] = i0
            i1_inds[i_node] = i1
            less_inds[i_node] = -1
            more_inds[i_node] = -1
            split_dims[i_node] = -1
            split_values[i_node] = 0.0
        else:
            # Split on largest dimension
            lengths = hi - lo
            split_dim = np.argmax(lengths)
            split_value = lo[split_dim] + 0.5 * lengths[split_dim]

            # Partition i0:i1 range into points where points[i, split_dim] < split_value
            i = i0
            j = i1 - 1

            while i < j:
                while i < j and points[i, split_dim] < split_value:
                    i += 1
                while i < j and points[j, split_dim] >= split_value:
                    j -= 1

                # Swap points
                if i < j:
                    for d in range(dimension):
                        temp = points[i, d]
                        points[i, d] = points[j, d]
                        points[j, d] = temp

                    temp_i_node = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp_i_node

            if points[i, split_dim] < split_value:
                i += 1

            i_split = i

            # Now it holds that:
            # for i in range(i0, i_split): assert(points[i, split_dim] < split_value)
            # for i in range(i_split, i1): assert(points[i, split_dim] >= split_value)

            # Ensure that each node has at least min_leaf_size children
            i_split = max(i0 + min_leaf_size, min(i1 - min_leaf_size, i_split))

            less = n_nodes
            n_nodes += 1
            more = n_nodes
            n_nodes += 1

            # push i0, i_split, less
            stack[stack_size] = i0
            stack_size += 1
            stack[stack_size] = i_split
            stack_size += 1
            stack[stack_size] = less
            stack_size += 1

            # push i_split, i1, more
            stack[stack_size] = i_split
            stack_size += 1
            stack[stack_size] = i1
            stack_size += 1
            stack[stack_size] = more
            stack_size += 1

            i0_inds[i_node] = i0
            i1_inds[i_node] = i1
            less_inds[i_node] = less
            more_inds[i_node] = more
            split_dims[i_node] = split_dim
            split_values[i_node] = split_value

    return n_nodes


@njit("void(i8[:], i8[:], i8[:], i8[:], i8[:], f4[:, :, :], f4[:], f4[:, :], f4[:, :], i8[:, :], f4[:, :], i8)", cache=True, nogil=True, parallel=True)
def _find_knn(
    i0_inds,
    i1_inds,
    less_inds,
    more_inds,
    split_dims,
    bounds,
    split_values,
    points,
    query_points,
    out_indices,
    out_distances,
    k,
):
    dimension = points.shape[1]

    # For each query point
    for i_query in prange(query_points.shape[0]):
        query_point = query_points[i_query]

        distances = out_distances[i_query]
        indices = out_indices[i_query]

        # Expect log2(len(points) / min_leaf_size) depth, 1000 should be plenty
        stack = np.empty(1000, np.int64)

        n_neighbors = 0
        stack[0] = 0
        stack_size = 1

        # While there are nodes to visit
        while stack_size > 0:
            stack_size -= 1
            i_node = stack[stack_size]

            # If we found more neighbors than we need
            if n_neighbors >= k:
                # Calculate distance to bounding box of node
                dist = 0.0
                for d in range(dimension):
                    p = query_point[d]
                    dp = p - max(bounds[i_node, 0, d], min(bounds[i_node, 1, d], p))
                    dist += dp * dp

                # Do nothing with this node if all points we have found so far
                # are closer than the bounding box of the node.
                if dist > distances[n_neighbors - 1]:
                    continue

            # If leaf node
            if split_dims[i_node] == -1:
                # For each point in leaf node
                for i in range(i0_inds[i_node], i1_inds[i_node]):
                    # Calculate distance between query point and point in node
                    distance = 0.0
                    for d in range(dimension):
                        dd = query_point[d] - points[i, d]
                        distance += dd * dd

                    # Find insert position
                    insert_pos = n_neighbors
                    for j in range(n_neighbors - 1, -1, -1):
                        if distances[j] > distance:
                            insert_pos = j

                    # Insert found point in a sorted order
                    if insert_pos < k:
                        # Move [insert_pos:k-1] one to the right to make space
                        for j in range(min(n_neighbors, k - 1), insert_pos, -1):
                            indices[j] = indices[j - 1]
                            distances[j] = distances[j - 1]

                        # Insert new neighbor
                        indices[insert_pos] = i
                        distances[insert_pos] = distance
                        n_neighbors = min(n_neighbors + 1, k)
            else:
                # Descent to child nodes
                less = less_inds[i_node]
                more = more_inds[i_node]
                split_dim = split_dims[i_node]

                # First, visit child in same bounding box as query point
                if query_point[split_dim] < split_values[i_node]:
                    stack[stack_size] = more
                    stack_size += 1
                    stack[stack_size] = less
                    stack_size += 1
                else:
                    # Next, visit other child
                    stack[stack_size] = less
                    stack_size += 1
                    stack[stack_size] = more
                    stack_size += 1

        # Workaround for https://github.com/numba/numba/issues/5156
        stack_size += 0

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
