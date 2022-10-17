## TextGenerationWithLSTM

This project is about implementing Language Modelling using an LSTM based Recurrent Neural Network. “Aesop’s Fables (A Version for Young Readers)” from [Project Gutenberg](https://www.gutenberg.org) was used as the dataset. The LSTM model is supposed to generate text based on the input sentences provided.

## Motivation
This project was a part of Assignment 3 of the "Deep Learning Lab" course at USI, Lugano taken by [Dr. Kazuki Irie](https://people.idsia.ch/~kazuki/).

## Tech used
<b>Built with</b>
- [Python3](https://www.python.org)
- [NumPy](https://numpy.org)
- [PyTorch](https://pytorch.org)


## Features
The project includes implementation of the following concepts:
- Text chunking to facilitate Truncated Backpropogation Through Time (BPTT)
- LSTM language model with  input embedding layer, multiple LSTM layers using nn.LSTM, and a final softmax classification layer to predict the next character
- Greedy decoding algorithm
- Random sampling during decoding
- Monitoring Perplexity ($p$) of the model while training

$$ p = \exp \left(-\frac{1}{N} \sum_{n=1}^{N}\log p(w_n|w_0^{n-1}) \right) $$

A report can be found explaining my findings in this repo titled `Report.pdf`