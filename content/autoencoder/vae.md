+++
author = "C.J. Duan"
title = "scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI"
date = "2023-11-07"
description =  "scGPT"
math = true
+++


by Haotian Cui, Chloe Wang,  Hassaan Maan,  Bo Wang

___
**Keywords**: generative pre-trained transformer, foundation models for cell and gene biology, multi-batch integration, multi-omic integration, cell-type annotation, genetic perturbation prediction, and gene network inference 
___
[Abstract](https://www.biorxiv.org/content/10.1101/2023.04.30.538439v1) 

Generative pre-trained models have achieved remarkable success in various domains such as natural language processing and computer vision. Specifically, the combination of large-scale diverse datasets and pre-trained transformers has emerged as a promising approach for developing foundation models. While texts are made up of words, cells can be characterized by genes. This analogy inspires us to explore the potential of foundation models for cell and gene biology. By leveraging the exponentially growing single-cell sequencing data, we present the first attempt to construct a single-cell foundation model through generative pre-training on over 10 million cells. We demonstrate that the generative pre-trained transformer, scGPT, effectively captures meaningful biological insights into genes and cells. Furthermore, the model can be readily finetuned to achieve state-of-the-art performance across a variety of downstream tasks, including multi-batch integration, multi-omic integration, cell-type annotation, genetic perturbation prediction, and gene network inference. 

***
[GitHub](https://github.com/bowang-lab/scGPT).
____

![](https://www.biorxiv.org/content/biorxiv/early/2023/05/01/2023.04.30.538439/F1.large.jpg?width=800&height=600&carousel=1)
____

@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
___
$$
[z_\mu,z_\sigma]^T = e(\eta_e,x)
$$

$$Z \sim N(z_\mu, \sigma^2_zI)$$
$$\hat{x} = d(\eta_d,Z)$$