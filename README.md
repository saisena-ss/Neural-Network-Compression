# Neural-Network-Compression

One of the problems of deploying deep neural networks in end devices with limited memory is size of the network which can take up significant space and also increase the test time during inference. One way to solve this problem is to compress the network using singular value decomposition. By decomposing the each layer's weight matrices at every epoch, low rank approximation of weights can be achieved without significant loss in performance reducing the dimensionality of the weights therefore reducing the size of the network.
