+++
title = "Markov Logic Networks"
#slug = "contact"
date = "2022-09-30"
math = true
+++

by Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li
___
Robust learning with reasoning, Markov logic network, graph convolutional network, certified robustness, vari-ational inference
___
**[Abstract](https://doi.org/10.48550/arXiv.2209.05055)**

Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions (e.g., whether the input is furry), and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning (e.g., IsPanda =) IsFurry). However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets such as AwA2, Word50, GTSRB, and PDF malware, and we show that CARE achieves significantly higher certified robustness (e.g., the certified accuracy is improved from 36:0% to 61:8% under `2 radius 2:0 on AwA2) compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration
___
 