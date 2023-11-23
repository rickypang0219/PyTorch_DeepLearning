# Expectation Maximization (EM) Algorithm 


## Gaussian Mixture Model (GMM)
The Gaussian Mixture Model is an example of generatice algorithm with hidden/latent variable. In generative algorithms we mean that we aim at modelling the distribution $
P(X | y)$, rather than $P(y | X)$ like usual discriminative algorithms do, for instance, linear and logistic regression. 

In supervised learning, the dataset is usually given by $(X,y)$. For example, in Gaussian discriminative analysis (GDA), $X$
can be a set of dog/cat pictures and $y$ is the corresponding labellings (e.g. 0/1). Then, by assuming $X| y={0/1} \sim \mathcal{N}(\mu_{0/1}, \Sigma)$ is normal distributed and $y \sim Bern(\phi)$, we can train the model by MLE  

$$
\begin{align}
l(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i} P(X^{i}, y^{i} ; \phi, \mu_0, \mu_1, \Sigma ) \\ 
&= \log \prod_{i} P(X^{i} | y^{i} ; \mu_0, \mu_1, \Sigma ) P(y^{i} ; \phi) \\ 
\end{align}
$$

On the contrary, in unsupervised learning setting the dataset is given by $X$ solely and we try to learn something new from the data itself. Therefore, GMM is obtained by optimizing the MLE

$$
\begin{align}
l(\phi, \mu_0, \mu_1, \Sigma) &= \sum_i \log  P(X^{i} ; \phi, \mu_0, \mu_1, \Sigma ) \\ 
&= \sum_i  \log \sum_{\{z\}}^{k} P(X^{i} | z^{i} ;\mu, \Sigma ) P(z^{i} ; \phi) \\ 
\end{align}
$$

where $z_i$ is the hidden variables. 



## Expectation Maximization of GMM 


# References
- [Andrew Ng CS229 Lecture 14 EM algorithm](https://www.youtube.com/watch?v=rVfZHWTwXSA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=14)
- [EM Aklgorithm and how it work](https://www.youtube.com/watch?v=REypj2sy_5U&t=1s)
- [Gaussian Mixture model Medium post](https://medium.com/swlh/gaussian-mixture-models-gmm-1327a2a62a)
- [What is the expectation maximization algorithm?](http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf)
