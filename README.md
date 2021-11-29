# K-Means Unsupervised Clustering

description

# Installation

description

# Testing

description

# Analysis

For analysis of the iris dataset, use the below bash command:

```
for num_cluster in {1..140}; do for ((seed=0; seed<100; seed++)); do echo "cat data/iris-data.txt | ./split.bash 10 python kmeans.py $seed $num_cluster --percentage True --precision 3"; done | ./parallelize.bash; done >> stats/iris_out.txt
```

For analysis of cancer dataset, use the below bash command:

```
for num_cluster in {1..95}; do for ((seed=0; seed<100; seed++)); do echo "cat data/cancer-data.txt | ./split.bash 10 python kmeans.py $seed $num_cluster --percentage True --precision 3"; done | ./parallelize.bash; done >> stats/cancer_out.txt
```

# Future Work

description
