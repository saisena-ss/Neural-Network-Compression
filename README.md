# Neural-Network-Compression

### Introduction
Deploying deep neural networks on end devices with limited memory can be challenging due to the size of the network, which not only consumes significant storage space but also increases inference time. One effective solution to mitigate these issues is to compress neural networks using Singular Value Decomposition (SVD). In this project, I explore the concept of network compression through SVD on the popular MNIST dataset.

### Problem Statement
Deep neural networks often have millions of parameters, making them large and resource-intensive. This can be problematic when deploying them on edge devices such as mobile phones or embedded systems, which have limited memory and computational resources. Compressing neural networks while preserving their performance is crucial to make them more suitable for these resource-constrained environments.

### Solution
Singular Value Decomposition (SVD) is a matrix factorization technique that can be used to compress neural networks. The basic idea is to decompose each layer's weight matrices into three separate matrices: U, Σ (Sigma), and V^T, where U and V^T are orthogonal matrices, and Σ is a diagonal matrix containing the singular values. By retaining only the top-k singular values and their corresponding columns from U and rows from V^T, one can achieve a low-rank approximation of the weight matrices. This low-rank approximation reduces the dimensionality of the weights, thereby reducing the size of the network.

### Implementation
1. Dataset
I used the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9). This dataset is widely used for digit classification tasks and serves as an excellent benchmark for this project.

2. Neural Network Architecture
I trained a deep neural network with five hidden layers with 1024 units.

3. SVD Compression
At every epoch during the training process, I applied SVD to the weight matrices of each layer. The steps involved in the SVD compression process were as follows:

Decompose the weight matrix into U, Σ, and V^T using numpy's SVD function.
Retain the top-k singular values from Σ.
Truncate U and V^T accordingly to create a low-rank approximation of the weight matrix.
Replace the original weight matrix with the compressed version.
Continue training the network.
4. Evaluation
I evaluated the compressed network's performance on the MNIST test dataset by measuring accuracy and comparing it to the accuracy of the original uncompressed network. The goal was to achieve significant compression without a significant drop in accuracy.

### Results
By compressing the model from 1024 units to low rank of 20, I achieved substantial compression while maintaining a satisfactory level of accuracy. The exact value of k may vary depending on the specific network architecture and dataset.

### Conclusion
Neural network compression using Singular Value Decomposition is a valuable technique to reduce the size of deep neural networks, making them more suitable for deployment on resource-constrained devices. By carefully applying SVD at each epoch during training, I can achieve significant compression without sacrificing performance. This approach can be adapted to various neural network architectures and datasets, providing a versatile solution for optimizing deep learning models.
