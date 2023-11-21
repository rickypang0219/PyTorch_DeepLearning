# Expectation Maximization (EM) Algorithm 


In the Gaussian mixture model (GMM), we dont know whether the data comes from which gaussian, unlike Gaussian discriminative analysis (GDA). In GDA, we know the categories of the data belonging to and we can train the model based on conditional probablilty, simply asking 

$$
P(B \vert x ) = \frac{ P(x \vert B ) } { P(x \vert B) P(B) +  P(x \vert A) P(A) } 
$$

in which we assume that there are two categories $A$ and $B$. The likelihood $P(x \vert B)$ is given by normal distribution with mean $\mu$ and s.d. $\sigma$ 

$$
x \vert B \sim \mathcal{N} (\mu, \sigma)
$$

This is the setting of GDA. However, GMM does not come with the labelling of the categories, meaning that the input is features solely. Therefore, the categories in GMM is treated as the 'hidden variables' of the model and the goal of GMM is finding out these hidden varibales for classifying which cluster does the data belong to. 