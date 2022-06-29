# CosFace: Large Margin Cosine Loss for Deep Face Recognition
Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song. _3 Apr 2018_

The traditional softmax loss of deep CNNs usually lacks the power of discrimination. To address this problem, recently several loss functions such as center loss, large margin softmax loss, and angular softmax loss have been proposed. All these improved losses share the same idea: maximizing inter-class variance and minimizing intra-class variance.
This paper, we propose a novel loss function, namely large margin cosine loss (LMCL), to realize this idea from a different perspective.

* Official paper: [arXiv]((https://arxiv.org/abs/1801.09414))
* Official code: [Github](https://paperswithcode.com/paper/cosface-large-margin-cosine-loss-for-deep)

## Overview

1. Drawback of softmax loss
2. Large Margin Cosine Loss (CosFace)
3. Hyperparameter tuning

## I. Drawback of softmax loss
>  The softmax loss separates features from different classes by maximizing the posterior probability of the ground-truth class.

- The softmax formulated as:
  $$ L_{s}$ = $\frac{1}{N} \sum_{i=1}^n - log\frac{e^{f_{y_i}}}{\sum_{j=1}^C e^{f_j}} $$
  we fix the bias is 0 then $f_j$ is given by:
   $$f_i = W_{j}^T .x = ||W_j|| ||x|| \cos $\theta$_j $$
where $\theta$ is the angle between $W_j$ and $x$. 
-  To develop effective feature learning, the norm of W
should be necessarily invariable. Thus, we fix $||W_j||$ = 1 by $L_2$ normalization and fix $||x||$ = $s$. Then softmax will be only depended on $\theta$ as:
  $$ L_{norm\_s}$ = $\frac{1}{N} \sum_{i=1}^n - log\frac{e^{s \cos(\theta_{(y_i, i)})}}{\sum_{j=1}^C e^{s \cos(\theta_{(y_i, i)})}} $$

### Explain
* sentence 1
* code (maybe)



## II. Large Margin Cosine Loss (CosFace)
## III. Hyperparameter tuning

