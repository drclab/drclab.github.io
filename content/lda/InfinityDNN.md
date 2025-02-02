+++
title = "Variational Inference for Infinitely Deep Neural Networks"
#slug = "contact"
date = "2022-09-01"
math = true
+++

by  Achille Nazaret, David Blei
___
**Keywords**:  
___
**[Abstract](https://doi.org/10.48550/arXiv.2209.10091)**

 We introduce the unbounded depth neural network (UDN), an infinitely deep probabilistic model that adapts its complexity to the training data. The UDN contains an infinite sequence of hidden layers and places an unbounded prior on a truncation L, the layer from which it produces its data. Given a dataset of observations, the posterior UDN provides a conditional distribution of both the parameters of the infinite neural network and its truncation. We develop a novel variational inference algorithm to approximate this posterior, optimizing a distribution of the neural network weights and of the truncation depth L, and without any upper limit on L. To this end, the variational family has a special structure: it models neural network weights of arbitrary depth, and it dynamically creates or removes free variational parameters as its distribution of the truncation is optimized. (Unlike heuristic approaches to model search, it is solely through gradient-based optimization that this algorithm explores the space of truncations.) We study the UDN on real and synthetic data. We find that the UDN adapts its posterior depth to the dataset complexity; it outperforms standard neural networks of similar computational complexity; and it outperforms other approaches to infinite-depth neural networks.
___
[Github](https://github.com/anazaret/unbounded-depth-neural-networks)
