# Part-based Pseudo Label Refinement for Unsupervised Person Re-identification
Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon. _28 Mar 2022_
>  In this paper, we propose a novel Part-based Pseudo Label Refinement (PPLR) framework that reduces the label noise by employing the complementary relationship between global and part features. Specifically, we design a cross agreement score as the similarity of k-nearest neighbors between feature spaces to exploit the reliable complementary relationship. 
> Based on the cross agreement, we refine pseudo-labels of global features by ensembling the predictions of part features, which collectively alleviate the noise in global feature clustering. We further refine pseudo-labels of part features by applying label smoothing according to the suitability of given labels for each part. Thanks to the reliable complementary information provided by the cross agreement score, our PPLR effectively reduces the influence of noisy labels and learns discriminative representations with rich local contexts. 

* Official paper: [arXiv](https://arxiv.org/abs/2203.14675)
* Official code: [Github](https://github.com/yoonkicho/pplr)

## Overview

1. Related work
2. Architecture
3. Ablation Study

## 1. Related work
- Learning with noisy labels
  - Loss adjustment approaches to design a robust loss function against the label noise as mean absolute error (MAE) loss, Generalized crossentropy (GCE), symmetric crossentropy (SCE) loss
  - the sample re-weighting scheme based on the reliability of a given label
  -  These loss functions, however, are designed for simple image classification tasks and are not suitable for open-set person re-ID tasks.
  
- Part-based approaches for person re-ID
  - Fine-grained information on human body parts( human parsing, pose estimation, attention mechanism) is an essential clue to distinguish people
  - Recent methods  utilize part features to exploit robust feature similarity for accurate pseudo-labels.   

- Unsupervised approaches for person re-ID
  -  unsupervised domain adaptation (UDA)
  -  unsupervised learning depending on whether an external labeled source domain data is used
  -  Clustering-based methods apply the contrastive learning scheme with cluster proxies

## 2. Architecture
There are 2 stages: 
   -  the clustering stage: extract global and part features and assign pseudo-labels through global feature clustering. Then compute a cross agreement score for each sample based on the similarity between k-nearest neighbors of global and part features
   -  the training stage  : mitigate the label noise using the proposed pseudo-label refinement methods based on the cross agreement( agreement-aware label smoothing for part feature, part-guided label refinement for global feature)
  
![fig](../../asset/images/PPLR/fig1.png#center)

### 2.1 Part-based Unsupervised re-ID Framework
> present a part-based unsupervised person reID framework that utilizes fine-grained information of the part features.  Use both the global and part features to represent an image.

- step 1: model first extracts the _shared representation_ $F_{\theta}(x_i) \in R^{C*H*W}$ where _C, H, W_ are channel, height, width of the **feature map**. 
  - Using GAP over feature map ==> **Global feature** $f_i^g$
  - Dividing horizontally(H' direction) to _N parts_ and applying AP ==> **local features**(part features) $\{f_i^{p_n}\}_{n=1}^{N_p}$ with shape $R^{C*\frac{H}{N_p}*W}$

- step 2: Generating the pseudo-labels based on clustering results. 
  - Perform DBSCAN clustering on the globel feature set $\{f_i^g\}_{i=1}^N$ with _N is number samples_.
  - use the cluster assignment as pseudo-labels => $(x_i, y_i)_{i=1}^N | y_i \in R^K$ where K is the number of clusters.
- step 3: Compute loss funtion:
  - global cross-entropy loss: $L_{gce} = - \sum_{i=1}^{N_D} y_i . log(q_i^g)$
    where $q_i^g = h_{\phi_g}(f_i^g) \in R^K$ is the prediction vector by global feature, and _h_ is the global feature classifier.
  - local(part) cross entropy: $L_{pce} = - \frac{1}{N_p} \sum_{i=1}^{N} \sum_{n=1}^{N_p} y_i . log(q_i^{p_n})$
    where $q_i^{p_n} = h_{\phi_g}(f_i^{p_n}) in R^K$ is the prediction vector by _n_-th part feature space $p_n$, and _h_ is the classifier for the part feature space. 
  - softmax-triplet loss:
  $$L_{softTriplet =  - \sum_{i=1}^{N} log (\frac{e^{||f_i^g - f_{i,n}^g||}}{e^{||f_i^g - f_{i,p}^g||} + e^{||f_i^g - f_{i,n}^g||}}) $$
  where ||.|| denotes the L2-norm, the subcripts _(i,p)_ and _(i,n)_ respectively the hardest positive and negative samples of the image $x_i$ in mini-batch.
  - *Optional loss*: camera-aware proxy to improve the discriminability across camera views. This loss attemp _pull_ together the proxies are within the same cluster but in different cameras, _reduce_ the intra-class variance caused by disjoint camera views.
    - compute the camrera-aware proxy $c_{a,b}$ as the cenntroid of the features that have _same camera label_ **a** and _same cluter(plabel)_ **b**
    - with $P_i & Q_i$ are the index sets of the positive and hard negative camera-aware proxies for $f_i^g$, the inter-camera contrastive loss as:
    $$L_{cam} = - \sum_{i=1}^N \frac{1}{|P_i|} \sum_{j \in P_i} log \frac{exp(c_j^\tau f_i^g / \tau)}{\sum_{k \in P_i \cup Q_i } exp(c_k^\tau f_^g / \tau)}$$ 
  
  **Final objective**
  $$L = L_{gce} + L_{pce} + L_{softTriplet} + \lambda L_{cam}$$


### 2.2 Cross Agreement
> a cross agreement score that captures how reciprocally similar the k-nearest neighbors of global and part features are for refining pseudo-labels of global features

  - the cross agreement score is defined as the Jaccard similarity between the knearest neighbors of the global and part features.

**STEPS**
  - to perform a KNN search on the gobal and each local feature spaces independently to produce $(1+N_p)$ ranked lists on each image.
  - Compute the cross agreement score between the global feature space _g_ and the _n_-th parth feature space $p_n$ for each _i_-th image by:
    $$C_i(g, p_n) = \frac{R_i(g,k) \cup R_i(p_n,k)}{|R_i(g,k) \cap R_i(p_n,k|} \in [0,1]$$
  where $R_i(g,k) & R_i(p_n,k)$ are the sets of the indices for top-k samples in the ranked list.
  - higl cross agreement implies the speudo label have high reliable complementary information.

### 2.3 Pseudo Label Refinement
Based on the cross agreement scores, we alleviate the pseudo label noise by considering:
* whether the pseudo-labels by global feature clustering are suitable for each part feature
* whether the predictions of part features are appropriate for refining pseudo-labels of global features

- **Agreement-aware label smoothing.**
  

- **Part-guided label refinement.**  


- **Overall training objective.** 


## 3. Ablation Study


