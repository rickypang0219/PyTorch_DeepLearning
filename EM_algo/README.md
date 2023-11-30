# Expectation Maximization (EM) Algorithm 


## 1. Gaussian Mixture Model (GMM)
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



## 2. Expectation Maximization 

### 2.1 Evidence of the Lower Bound (ELBO)
The evidence of Lower Bound (ELBO) is an important concept in variational inference since it gives the lower bound of the lost function. By construting the lower bound, we can successively find out the local maximum of the lost function. Throughout the derivation, we assume there is only one example of $x$ for simplicity. Therefore, the log-likelihood can be written as 

$$
\log p(x ; \theta) = \log \sum_z p(x, z ; \theta)
$$

by the marginalization. Let $Q(z)$ be a distribution over $z$ satisfying $\sum_z Q(z)  = 1$, we consider the following 

$$
\begin{align}
\log p(x; \theta) &= \log \sum_z p(x, z ; \theta) \\ 
&= \log \sum_z  Q(z)  \frac{p(x, z ; \theta) }{Q(z)} \\ 
& \geq  \mathbb{E}_{z \sim Q} \big \{ \log \frac{p(x, z ; \theta) }{Q(z)} \big \}
\end{align} 
$$



where in the last step we use the Jensen inequlity for $\log$ function. The above derivation shows that for a log-likelihood $\log p(x ; \theta)$, there exists a corresponding lower bound for **any** possible distribution $Q(z)$. Therefore, we define the evidence of lower bound (ELBO) as 
$$
\text{ELBO}(x; Q, \theta) = \mathbb{E}_{z \sim Q} \Big \{ \log \frac{p(x, z ; \theta) }{Q(z)} \Big \} , \\ 
\forall Q, \theta, x , ~ \log p(x;\theta) \geq ELBO(x; Q, \theta).
$$

In some texts, the ELBO maybe re-written as 

$$
\begin{align}
\text{ELBO}(x; Q, \theta) &= \mathbb{E}_{z \sim Q} \Big \{ \log \frac{p(x, z ; \theta) }{Q(z)} \Big \} \\ 
 &= \mathbb{E}_{z \sim Q} \Big \{ \log \frac{p(x, z ; \theta) p(z\vert x ; \theta) }{Q(z) p(z\vert x ; \theta)} \Big \} \\ 
&= E_{z \sim Q} \Big[ \log p(x; \theta) - \log \frac{Q(z)}{p(z \vert x ;\theta)} \Big]  \\ 
&= \log p(x; \theta) - E_{z \sim Q} \log \frac{Q(z)}{p(z \vert x ;\theta)}   \\ 
&= \log p(x; \theta) - D_{KL}(Q(z) ~  \Vert ~ p(z \vert x))
\end{align}
$$

Having a lower bound is not telling. What we want to have is a tight bound, meaning that the equality hold given $\theta$. One interesting observation of the alternative ELBO form is that it consists of the log-likelihood and the KL Divergence term. Because of $D_{KL} \geq 0$, when we are maximizing the log-likelihood, $D_{KL}$ vanishes. Therefore, from the alternative form of ELBO, we can find out $Q(z)$ as 

$$
Q(z) = p(z \vert x ; \theta )
$$

To summarise, by maximizing the ELBO, we implicitly maximizing the log-likelihood. 
### 2.2 Expectation/ Maximization Step

In the expectation step, we propose 

$$
Q(z) =  p(z \vert x ; \theta )
$$

Once we have $Q(z)$, we can plug this into the definition of ELBO. The next step is maximization of the ELBO, we hope to find the model parameter $\theta$ by maximization 

$$
\hat \theta = \text{arg max}_{\theta} ~ \text{ELBO}(x;  Q,\theta)
$$





##  References
- [Andrew Ng CS229 Lecture 14 EM algorithm](https://www.youtube.com/watch?v=rVfZHWTwXSA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=14)
- [EM Aklgorithm and how it work](https://www.youtube.com/watch?v=REypj2sy_5U&t=1s)
- [Gaussian Mixture model Medium post](https://medium.com/swlh/gaussian-mixture-models-gmm-1327a2a62a)
- [What is the expectation maximization algorithm?](http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf)
