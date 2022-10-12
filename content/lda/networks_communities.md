+++
title = "Efficient discovery of overlapping communities in massive networks"
#slug = "contact"
date = "2012-12-01"
math = true
+++

by    Prem K. Gopalan and David M. Blei
___
**Keywords**:  network analysis, Bayesian statistics, massive dat
___
**[Abstract](https://doi.org/10.1073/pnas.1221839110)**

Detecting overlapping communities is essential to analyzing and exploring natural networks such as social networks, biological networks, and citation networks. However, most existing approaches do not scale to the size of networks that we regularly observe in the real world. In this paper, we develop a scalable approach to community detection that discovers overlapping communities in massive real-world networks. Our approach is based on a Bayesian model of networks that allows nodes to participate in multiple communities, and a corresponding algorithm that naturally interleaves subsampling from the network and updating an estimate of its communities. We demonstrate how we can discover the hidden community structure of several real-world networks, including 3.7 million US patents, 575,000 physics articles from the arXiv preprint server, and 875,000 connected Web pages from the Internet. Furthermore, we demonstrate on large simulated networks that our algorithm accurately discovers the true community structure. This paper opens the door to using sophisticated statistical models to analyze massive networks.
 
___
[Github](https://github.com/premgopalan/svinet)
