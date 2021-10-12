# Mnist : What is the Mnist?

  ## Let claclute each hadwritten 0 to 9's pictures pixel using Mnist datasets.
  
  Datasets address: 
  
  1. This data consists of 28×28, total 784.
  2. mean and variance claculate 
  3. Eigenvalue and Eigenvecor 
  
 

## Make define 

### Rule....

we haven't to use librarryies (Sk, kreas etc...)
Implement the kMeans algorithm and run it on the MNIST ‘3’ and ‘9’ images together.  

This means no built-in libraries are allowed for computing the kMeans algorithm (no scipy, scikit-learn or similar). You have to code the algorithm from scratch. Basic libraries like numpy, pandas, math, csv, etc. that are not related to computing kMeans automatically are allowed.


Try with raw image and 2,5,10 dim eigenspaces, each of which you can try different number of clusters, (k = 2, 3, 5, 10.)  In total, you need to run kMeans 16 times (4 dimension cases x 4 number of clusters). 
'raw image and 2, 5, 10 dim eigenspaces' is related to the dataset. This means that first, you have to apply kMeans to the dataset (of 9s and 3s) without any pre-processing. Then you have to pre-process the dataset by reducing its dimensionality. This is done by using eigenspace projections. The dimensionalities you have to try are 2, 5, and 10.

Once you have the four datasets, for each one of them, run the kMeans algorithm with different parameters k (k = 2, 3, 5, 10.) 


Submit a zip file including your code and report (in pdf) about your implementation and the experiments. 

In the report, you can use any resource you want to explain your implementation and results. For small dimensions, you can plot the grouped clusters, or you can plot some image samples from each resulting cluster to show how the algorithm groups them, etc. You can also check the intra-cluster variation if you want to, and report about it. For high dimensions, you might not be able to visualize the clustering by plotting. You can use any resource you want to explain the results in these cases.

