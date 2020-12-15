# Parallel KMeans for Image segmentation  
## A set of parallel implementations of an Euclidean based KMeans  

### Overview  
The basic KMeans algorithm implemented takes as input an image of size n x m x c, which corresponds
respectively to the height, width and number of channels, plus a number of centroids k. The complexity of the computations is thus O(n.m.c.k) for each cycle. The sequential implementation is a succession of nested loops that compute distance from each of the k centroids to the n x m pixels in the image, before determining the minimum, computing the new barycenter and updating the centroids.

Two stopping conditions were provided: **max iterations** *(which reflects the number of cycles allowed)* and **epsilon** which serves as a threshold for centroids displacement after new cluster affectation and barycenter
computation.

### I. OpenMP Implementation  
Given that the input is generally an image with a resolution higher than 100x100, the parallelization can be done using a **static scheduling over rows** alone. Thus, the two main methods that can efficiently take advantage of this are the **centroids computation** *(distance, barycenter, update)* and **image segmentation** *(re-painting each pixel based on the previously computed mapping)*. Static block sizes (chunks) are chosen carefully to avoid false sharing problems when writing on and cache misses when reading from the contiguous shared variable *(mapping, image, etc..)*.

Moreover, shared variables that need to be updated separately on different threads *(new centroid sums and cluster sizes)* but still require the overall result are being reduced as arrays *(Made possible starting openMP v4.0)*.

### II. TBB Implementation  *(In progress)*
### III. MPI Implementation   *(In progress)*



-----
Author : Ouadie EL FAROUKI - Mines Paristech, France. 2020