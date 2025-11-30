# 1D K-Means Clustering

## Description

This is a K-Means clustering algorithm for grouping 1D data into clusters, implemented using Python.

\
The steps of the K-Means clusting algorithm are as follows:

1. Randomly choose centres (in this case as sample from inputs)
2. Assign inputs to clusters by minimum distance to cluster's centre
3. Move centres to be in the middle of the clusters as assigned
4. Repeat steps 2 and 3 until successive assignments result in the same clusters being assigned, meaning we have reached convergence

\
The following Python packages were used in this project:

* Numpy
* Matplotlib


## Files

There are two separate Python files for this project.

The first is the iterative file, which shows the iterative process of the K-Means algorithm at each step.

The second is the optimal file, which performs the K-Means clustering a set number of times for each number of clusters/centres in the given range, and saves an image with the optimal assignment for each number of clusters, based on the minimum sum distances of inputs to their cluster's centre.

## How to run the project

The application can be run locally by pulling the contents of this repository, running the command "pip install -r requirements.txt", and then either running the iterative Python file to see the algorithm at every step on a single iteration, or the optimal Python file to see the optimal assignments and minimum variance found for each number of clusters after a number of iterations.