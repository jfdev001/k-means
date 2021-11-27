"""
Author: Jared Frazier
Project: OLA 4
File: kmeans.py
Class: CSCI 4350
Instructor: Dr. Joshua Phillips
Description: Script for unsupervised clustering of input data using
K-means algorithm.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import numpy as np


class Cluster:
    """Cluster for K-Means."""

    def __init__(self, centroid: np.ndarray = np.nan, label: int = None):
        """Define state for Cluster."""

        # Coordinates for cluster
        self.centroid = centroid

        # Label associated with cluster
        self.label = label


class KMeans:
    """K-Means Clustering Algorithm."""

    def __init__(
            self,):
        """Define state for KMeans.

        :return: None
        """

        return

    def train(
            self,
            num_clusters: int,
            training_data: np.ndarray) -> None:
        """Public method to train k-means.

        Calls `__train` method iteratively until the centroids
        no longer change. The algorithm proceeds by assigning each
        observation (row vector) to the cluster (randomly initialized)
        to the cluster with the nearest (Euclidean distance) mean.

        :param num_clusters:
        :param training_data:

        :return: None
        """

        # Extract data
        features = training_data[:, :-1]
        labels = training_data[:, -1]

        # Initialize clusters
        random_ixs = np.random.randint(
            low=0, high=training_data.shape[0], size=num_clusters)

        clusters = [Cluster(centroid=features[ix, :]) for ix in random_ixs]

        # Train clusters
        converged = False
        while not converged:

            # Copy the previous clusters
            prev_clusters = deepcopy(clusters)

            # Pass by reference the current clusters and modify them
            self.__train(
                features=features,
                labels=labels,
                clusters=clusters)

            # Check for convergence
            pass

        # Assign labels
        pass

    def test(
            self,
            testing_data: np.ndarray) -> np.ndarray[np.int64] or np.int64:
        """Public method to test using k-means centroids.

        :param testing_data:

        :return:
        """
        pass

    def __train(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            clusters: np.ndarray[Cluster]) -> None:
        """Private method for iteratively training centroids.

        For each sample in the dataset, determine the closest cluster
        and then assign 

        :param features:
        :param labels:
        :param clusters:

        16878152

        :return: None
        """
        pass

    def __assignment(self,):
        """"""
        pass

    def __update(self,):
        """"""
        pass

    def __check_convergence(self,):
        """"""
        pass

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
