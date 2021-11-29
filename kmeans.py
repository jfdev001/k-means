"""
Author: Jared Frazier
Project: OLA 4
File: kmeans.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Script for unsupervised clustering of input data using
K-means algorithm. 16878152
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import numpy as np


class Cluster:
    """Cluster for K-Means."""

    def __init__(self, centroid: np.ndarray[np.float64]):
        """Define state for Cluster."""

        # Coordinates for cluster
        self.centroid = centroid

        # Label associated with cluster
        self.label = None

        # Vectors from which the centroid will be calculated
        self.vectors = []

    def append_closest_vector(self, vector: np.ndarray[np.float64]) -> None:
        """Add a row vector to the list of vectors closest to centroid."""

        self.vectors.append(vector)

    def get_centroid(self, ) -> np.ndarray[np.float64]:
        """Get centroid attribute."""

        return self.centroid

    def get_label(self, ) -> np.float64:
        """Get label attribute."""

        return self.label

    def set_centroid(self,) -> None:
        """Use the vector list to compute the new centroid (mean along axis=0)"""

        # Set the centroid
        self.centroid = np.mean(self.vectors, axis=0)

    def set_label(self, label: int) -> None:
        """Set label attribute."""

        self.label = label

    def clear_closest_vectors(self,) -> None:
        """Clears the current list of vectors."""

        self.vectors = []

    def __eq__(self, other: Cluster) -> bool:
        """Check if centroids of two clusters are the same."""

        return np.all(np.equal(self.centroid, other.get_centroid()))

    def __sub__(self, other: np.ndarray[np.float64] or Cluster) -> np.float64:
        """Subtract Cluster with Cluster or with ndarray."""

        if isinstance(other, Cluster):
            return self.centroid - other.get_centroid()
        elif isinstance(other, np.ndarray):
            return self.centroid - other
        else:
            raise TypeError(':param other: is not a valid type.')

    def __repr__(self) -> str:
        """Information about a Cluster obj"""

        rep = f'{self.__class__} object at {hex(id(self))}:'
        rep += f' (centroid={self.centroid},'
        rep += f' label={self.label})'
        return rep


class KMeans:
    """K-Means Clustering Algorithm."""

    def __init__(self,):
        """Define state for KMeans."""

        # List of k-means clusters with the corresponding label
        self.clusters = None

    def train(
            self,
            num_clusters: int,
            training_data: np.ndarray[np.float64]) -> None:
        """Public method to train k-means.

        Calls `__train` method iteratively until the centroids
        no longer change. The algorithm proceeds by assigning each
        observation (row vector) to the cluster (randomly initialized)
        to the cluster with the nearest (Euclidean distance) mean.

        :param num_clusters: Number of clusters for k-means algorithm.
        :param training_data: Data with features and labels (last column).

        :return: None
        """

        # Extract data
        features = training_data[:, :-1]
        labels = training_data[:, -1]

        # Initialize clusters
        random_ixs = np.random.randint(
            low=0, high=training_data.shape[0], size=num_clusters)

        self.clusters = [Cluster(centroid=features[ix, :])
                         for ix in random_ixs]

        # Train clusters
        converged = False
        num_iter = 0
        while not converged:

            # Copy the previous clusters
            prev_clusters = deepcopy(self.clusters)

            # Pass by reference the current clusters and modify their centroids
            self.__assign_and_update_clusters(
                features=features,
                clusters=self.clusters)

            # Check for convergence
            converged = self.__check_convergence(
                prev_clusters=prev_clusters,
                clusters=self.clusters)

            num_iter += 1

        # LOGGING
        print(f'Convergence after `{num_iter}` iterations...')
        for cluster in self.clusters:
            print(cluster)

        # Assign labels
        pass

    def test(
            self,
            testing_data: np.ndarray) -> np.ndarray[np.float64] or np.float64:
        """Public method to test using k-means centroids.

        :param testing_data: Feature containing (continuous) data only.

        :return:
        """
        pass

    def __assign_and_update_clusters(
            self,
            features: np.ndarray[np.float64],
            clusters: list[Cluster]) -> None:
        """Private method for iteratively training centroids.

        For each sample in the dataset, determine the closest cluster,
        then assign each feature vector to the cluster and calculate
        the new cluster.

        :param features:
        :param clusters:

        :return: None
        """

        # Iterate through data set and assign feature vectors to clusters
        for feature_vector in features:

            # Compute euclidean distnace between vector and each centroid
            cluster_feature_dist_lst = []
            for cluster in clusters:
                euc_dist = self.__euclidean_distance(
                    arr_1=cluster, arr_2=feature_vector)

                cluster_feature_dist_lst.append(euc_dist)

            # Compute argmin of distances and then append the...
            # desired vector to the cluster class
            best_cluster_ix = np.argmin(cluster_feature_dist_lst)
            clusters[best_cluster_ix].append_closest_vector(feature_vector)

        # Update new centroids after all feature vectors have been
        # assigned to clusters... then clears the list of vectors
        # that the cluster currently tracks
        for cluster in clusters:
            cluster.set_centroid()
            cluster.clear_closest_vectors()

        # Pass by obj-ref, so return none
        return

    def __check_convergence(
            self,
            prev_clusters: list[Cluster],
            clusters: list[Cluster]) -> bool:
        """True if cluster centroids are all the same, false otherwise."""

        # Iterate through previous and current clusters and
        # determine whether any centroids are not the same
        all_clusters_are_same = True
        ix = 0
        while all_clusters_are_same and ix < len(clusters):

            # Get parallel clusters
            cur_cluster = clusters[ix]
            prev_cluster = prev_clusters[ix]

            # Compute whether the centroids are the same
            clusters_are_same = cur_cluster == prev_cluster

            # Update outer variable
            if not clusters_are_same:
                all_clusters_are_same = False

            # Update index var
            ix += 1

        # Result of cluster comparison
        return all_clusters_are_same

    def __test(self,):
        """"""
        pass

    def __euclidean_distance(
            self,
            arr_1: np.ndarray,
            arr_2: np.ndarray) -> np.float64:
        """Returns the 2-norm (euc. distance) of two arrays."""

        return np.linalg.norm(arr_1 - arr_2)


def cli(description: str) -> argparse.ArgumentParser:
    """Command line interface for script."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'random_seed',
        help='random seed for k-means centroid intialization.',
        type=int,)
    parser.add_argument(
        'num_clusters',
        help='number of clusters for k-means.',
        type=int,)
    parser.add_argument(
        'train_data_path',
        help='path to training data for k-means.',
        type=str,)
    parser.add_argument(
        'test_data_path',
        help='path to testing data for k-means.',
        type=str,)

    return parser


def main():

    # Set up CLI
    parser = cli('script for train-testing k-means algorithm')
    args = parser.parse_args()

    # Set the random seed
    np.random.seed(args.random_seed)

    # Get data
    training_data = np.loadtxt(fname=args.train_data_path)
    testing_data = np.loadtxt(fname=args.test_data_path)

    # Reshape if needed to row vector
    if len(training_data.shape) == 1:
        training_data = np.expand_dims(training_data, axis=0)
    if len(testing_data.shape) == 1:
        testing_data = np.expand_dims(testing_data, axis=0)

    # Instantiate k-means
    k_means = KMeans()

    # Train
    k_means.train(
        num_clusters=args.num_clusters,
        training_data=training_data)

    # Test
    preds = k_means.test(testing_data=testing_data)

    # Print testing outputs
    pass


if __name__ == '__main__':
    main()
