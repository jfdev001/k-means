# K-Means Unsupervised Clustering

The [K-Means Algorithm](https://en.wikipedia.org/wiki/K-means_clustering) is used to generated _K_ clusters with centroids that capture information about high dimensional data with (or without) labels. This repository uses labeled data to assess the performance of the algorithm, and essentially formulates the problem as a semi-supervised task.

# Installation

This repository makes use of the standard data analysis/scientific computing libraries:

`pip install numpy pandas matplotlib`

# Testing

To train and subsequently test the k-means algorithm, use the command line interface for `kmeans.py`

`python kmeans.py 0 3 data/iris-data.txt data/iris-data.txt`

For information about the arguments to any `.py` script, type

`python name_of_script.py -h`

# Analysis

A summary of results and analysis is in `report/report.pdf`; however, the commands to reproduce the figures are available below.

If running on Unix system, use `sed -i -e 's/\r$//' parallelize.bash` and `sed -i -e 's/\r$//' split.bash`. You will also need to make sure both `.bash` files are executable. This can be done with `chmod +x parallelize.bash` and `chmod +x split.bash`

For analysis of the iris dataset, use the below bash command:

```
for num_cluster in {1..140}; do for ((seed=0; seed<100; seed++)); do echo "cat data/iris-data.txt | ./split.bash 10 python kmeans.py $seed $num_cluster --percentage True --precision 3"; done | ./parallelize.bash; done >> stats/iris_out.txt
```

For analysis of cancer dataset, use the below bash command:

```
for num_cluster in {1..95}; do for ((seed=0; seed<100; seed++)); do echo "cat data/cancer-data.txt | ./split.bash 10 python kmeans.py $seed $num_cluster --percentage True --precision 3"; done | ./parallelize.bash; done >> stats/cancer_out.txt
```

# Future Work

The k-means algorithm spends significant time processing euclidean distances, so variations of the algorithm using caching and the triangle inequality could be used to accelerate the algorithm. Morever, different intialization strategies for the centroids could be used since the one employed for the current repo simply uses random initialization using a sample from the dataset (without replacement for _k>1_).
