## TextGenerationWithLSTM

This project is about implementing Language Modelling using an LSTM based Recurrent Neural Network. The LSTM model is trained on “Aesop’s Fables (A Version for Young Readers)” from [Project Gutenberg](https://www.gutenberg.org) to generate a story from a prompt provided by the user along the same lines as a story on Aesop’s Fables.

The model generates text recurrently by generating one character at a time according to its output distribution and feeding it back as an input to generate the next character.

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
- Random sampling decoding algorithm
- Monitoring Perplexity ($p$) of the model while training

$$ p = \exp \left(-\frac{1}{N} \sum_{n=1}^{N}\log p(w_n|w_0^{n-1}) \right) $$

## Data Prelimnaries
We use the “Aesop’s Fables (A Version for Young Readers)" text from [Project Gutenberg](https://www.gutenberg.org) to train our LSTM model. The project is desinged to generate a story from a prompt provided by the user along the same lines as a story on Aesop’s Fables.

The prompt can be somehting line "_The Farmer and the Stork_" which is not a title of any story in the text.

The dataset can be found in [data](./data/) folder. The text file `49010-0.txt` is the original text that is used with no preprocessing required. The text file has the following properties:

|       |  | 
| ----------- | ----------- | 
| Character vocabulary size      | 107       | 
| Number of lines   | 5033        | 
| Total number of words      | 32589       | 
| Number of characters   | 138881        | 
| Number of upper case character      | 9666       | 

## Vocabulary and Dataloader
The implementation of [Vocabulary](./src/vocabulary.py) class helps define the vocabulary of the text.

The [LongTextData](./src/text_dataset.py#L5) class prepares the text from source files to be tokenized with the help of the `Vocabulary` class that can be used by the LSTM model. 

We consider a very long string representing the whole book. Backpropagating gradients through an RNN which is enrolled as many times as there are characters in that long string (backpropagation through time; _BPTT_) will require too much memory. Instead, the string must be broken down into smaller text chunks. Chunking should be done such that the first token for the current chunk is the last token from the previous chunk. The first token of the chunk is the first input token to be fed to the language model, while the last token of the chunk is the last target token to be predicted by the model within a given chunk. We’ll then train the model by truncated backpropagation through time, i.e.,

- We limit the span of the backpropagation to be within one chunk. The length of the chunk is thus the BPTT span.
- We initialize the hidden state of the RNN at the beginning of the chunk by the last state from the previous chunk. We thus can not randomly shuffle chunks, since the RNN states need to be carried between consecutive text chunks.

This is done using [ChunkedTextData](./src/text_dataset.py#L49) class.

## Model
An [LSTMModel](./src/model.py) language model with:
- [Input embedding layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- Multiple LSTM layers using [nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)
- A final softmax classification layer to predict the next character

The model is initiated with [default configuration](./config/defaults.py) and dataset properties.

## Demo
A demo can be found in the [demo.py](./demo.py) script.
