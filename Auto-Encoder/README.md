# Auto-Encoder (AE)



# Variational Auto-Encoder (VAE)
Variational Auto-encoder is a auto-encoder with generative model. In AE we hope to find a latent representation of the data, which is the code $z$, by optimizing the reconstruction loss. However, the AE does not have ability for generating new images based on the input data. 

## Intractability of Likelihood/Posterior
In generative model in un-supervised learning such as Gaussian Mixture Model (GMM), we hope to model the distribution $
P(x; \theta)$ of the dataset. However, the distribution is **intractable** in most of the cases. If the distribution is intractable, it is hard for us to learn the parameters $\theta$ using MLE or MAP 

$$
\begin{align*}
\theta_{MLE} &= \arg \max_{\theta} \prod_{i} \sum_{z} P (x^{(i)} ,z  ; \theta) \\ 
\theta_{MAP} &= \arg \max_{\theta} \prod_{i} \sum_z P (x^{(i)} \vert \theta, z ) P(\theta) \\ 
\end{align*}
$$

Besides, the posterior distribution is also intractable hence we cannot apply EM algorithm to learn the parameters. Instead of finding the true posterior distribution $P(z \vert x ; \theta)$, we propose an recognition model to approximate it by $q_{\phi}(z \vert x)$, where $q_{\phi} (z \vert x)$. In terms of coding theory, we call $q_\phi (z \vert x)$ as an encoder since it generates a distribution over all possible values of code $z$ given a data-point $x$. Similarly, we can define $P(x \vert z)$ as the decoder. 

## ELBO / Variational Lower Bound 
When we are dealing with generative models, we use MLE to find the optimal parameters $\theta$ by maxmizing the log-likelihood

$$
\begin{align*}
\log P(x^{(i)} ; \theta ) &= \log \sum_z P(x^{(i)} ,z ; \theta )   \\ 
&= \log \sum_z q_{\phi} (z \vert x ) \frac{P(x^{(i)}, z ; \theta)}{q_\phi(z \vert x )}  \\ 
&\geq \mathbb{E}_{q_\phi(z \vert x)} \Big [\log \frac{P(x^{(i)} \vert z ; \theta) P(z ; \theta)}{q_\phi(z \vert x )} \Big ] \\ 
&= \mathbb{E}_{q_\phi(z \vert x)} \Big [\log P(x^{(i)} \vert z ; \theta)   \Big ] +   \mathbb{E}_{q_\phi(z \vert x)} \Big [ \log \frac{P(z ; \theta)}{q_{\phi}(z \vert x)} \Big ] \\ 
&= \mathbb{E}_{q_\phi(z \vert x)} \Big [\log P(x^{(i)} \vert z ; \theta)   \Big ]  - D_{KL} ( q_\phi(z \vert x)~  \Vert ~ P(z ; \theta))
\end{align*}
$$

and we call the variational lower bound or the evidence of lower bound (ELBO) as 

$$
\text{ELBO}(\phi, \theta ; x^{(i)}) = \mathbb{E}_{q_{\phi}(z \vert x)} \Big[\log P(x^{(i)} \vert z ; \theta) \Big]  - D_{KL} ( q_{\phi}(z \vert x)~  \Vert ~ P(z ; \theta))
$$

Having the definition of ELBO, we can find the corresponding parameters $\phi, \theta$ thats maximizing the log-likelihood by differentiating and optimizing the ELBO. However, using usual gradient estimator like Monte Carlo exhibits a large variance. Therefore, we need an alternative way to approximate the gradient estimator and this is done by reparameterization. 


## Reparamterization Trick 


