# Attention is all you need (Transformer)

# Introduction
Artificial Intelligence (AI) always affects human life, especially when ChatGOT arrives to public. ChatGPT is a great breakthrought in human histroy and natural language processing. The reason of why ChaTGPT is so power and impactful is because of self-atttention mechanism introduced in 'Attention is all you need'. In the following, we will discuss how self-attention mechanism revolutionize natural language processing field and computer vision.

# Why self-attention?
Before going straight to discuss what is self-attention mechanism, it is more essential to ask why do we need self-attention and why the scientists of Google Brain have to introduced this mechanism. In natural language process (NLP) and deep learning, captural the correlation between words are central task. For instance, given a long paragrah, we hope the learning algoritm can understand the correlation between different words. In the most extreme case, the algorithm should capture the first word and every other word in the whole paragraph. Traditional architecture like recurrent neural networks (RNNs) and long-short term memory (LSTM) can capture part of the whole correlation between words only. Besides, the training costs of such deep neural networks are expensive. These are the major difficulties before the introduction of self-attention. Therefore, self-attention aims at tackle these difficulties by computing the long-range correlation of inputs at the sametime parallely.

# What is self-attention?
Self-attention mechanism is an architecture to compute the correlation between different inputs (i.e. words represented in word embedding). In this


$$
Q,K,V = W^{q}I , W^{k}I, W^{v}I
$$

$$
\alpha = \text{softmax}( K^{T}Q / \sqrt{d_k})
$$

$$
O_{ij} = \sum_{k} (\alpha_{ik}{V}_{kj})
$$
