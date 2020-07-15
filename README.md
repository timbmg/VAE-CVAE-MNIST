# Variational Autoencoder & Conditional Variational Autoenoder on MNIST

VAE paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

CVAE paper: [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)

---
In order to run _conditional_ variational autoencoder, add `--conditional` to the the command. Check out the other commandline options in the code for hyperparameter settings (like learning rate, batch size, encoder/decoder layer depth and size).

---

## Results

All plots obtained after 10 epochs of training. Hyperparameters accordning to default settings in the code; not tuned.

### z ~ q(z|x) and q(z|x,c)
The modeled latent distribution after 10 epochs and 100 samples per digit.

VAE | CVAE
--- | --- 
<img src="https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/figs/1519649452.702026/E9-Dist.png" width="400"> | <img src="https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/figs/1519649461.195146/E9-Dist.png" width="400">

### p(x|z) and p(x|z,c)
Randomly sampled z, and their output. For CVAE, each c has been given as input once.

VAE | CVAE
--- | --- 
<img src="https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/figs/1519649452.702026/E9I937.png" width="400"> | <img src="https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/figs/1519649461.195146/E9I937.png" width="400">
