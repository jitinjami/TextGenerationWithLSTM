import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id
    
    def get_string(self, idx):
        assert(idx < len(self.id_to_string)), "Index must be stored in vocab"
        return self.id_to_string[idx]


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab
    

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches

def get_batches(batches: ChunkedTextData):
    '''
    Creates an iterable that generates x and y
    '''
    n_batches = len(batches) #87

    x = [None] * n_batches
    y = [None] * n_batches

    for i in range(n_batches):
        x = torch.transpose(batches[i][:-1], 0, 1) #Shape will be batch_size x sequential length: 32 x 64
        y = torch.transpose(batches[i][1:], 0, 1) #Shape will be batch_size x sequential length: 32 x 64
        yield x,y

class LSTMModel(nn.Module):

    def __init__(self, vocab_length, embedding_length, n_hidden=2048, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(num_embeddings=vocab_length, embedding_dim=embedding_length)


        self.lstm = nn.LSTM(embedding_length, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, vocab_length)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        r_output, hidden = self.lstm(embed, hidden)
        out = r_output.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

def predict(model, vocab, text, size, sampling = False):
    #Convert text to tensor using vocabulary
    model.eval()
    characters = [character for character in text]
    h = model.init_hidden(len(characters))

    for i in range(size):
        x = torch.tensor([[vocab.get_idx(characters[j]) for j in range(i,len(characters))]]).transpose(-1,0)
        y_pred, h = model(x, h)

        last_letter_logits = y_pred[-1]
        p = F.softmax(last_letter_logits, dim=0).detach().numpy()

        last_letter_index = np.argmax(p)
        if sampling:
            last_letter_index = np.random.choice(len(last_letter_logits), p=p)
        last_letter = vocab.get_string(last_letter_index)
        characters.append(last_letter)
    final_string = ''.join(characters)
    return final_string

def train(model, my_data, epochs = 100, batch_size = 32, seq_length = 64, lr = 0.001, clip = 1.0, device="cpu"):
    model.train()
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    data = my_data.data
    vocab = my_data.vocab

    train_batches = ChunkedTextData(data, batch_size, seq_length, pad_id=0)[:-1]

    model.to(device)


    for e in range(epochs):
        h = model.init_hidden(batch_size)

        for x_train, y_train in get_batches(train_batches):
            x_train.to(device)
            y_train.to(device)

            h = tuple([each.data for each in h])

            model.zero_grad()
            y_pred, h = model(x_train, h)
            
            loss = loss_fn(y_pred, y_train.reshape(batch_size*seq_length).long())
            perplex_loss = torch.exp(loss)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        
        print(predict(model, vocab, "Dogs like best to", 5), file=open("original.txt", "a"))

        model.train()

        print("Epoch: {}/{}...\n".format(e, epochs),
                "Training Loss: {:.4f}...\n".format(loss.item()),
                "Perplex Loss: {:.4f}...\n".format(perplex_loss), file=open("original.txt", "a"))


print("Original Parameters")
text_path = "49010-0.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
bptt_len = 64

my_data = LongTextData(text_path, device=device)
epochs = 50

n_hidden = 2048
n_layers = 1
vocab_length = len(my_data.vocab)
embedding_length = bptt_len

model = LSTMModel(vocab_length, embedding_length, n_hidden, n_layers)
print(model, file=open("original.txt", "a"))

train(model, my_data=my_data, epochs = epochs, batch_size = batch_size, seq_length=bptt_len, lr = 0.001, clip = 1.0, device=device)
torch.save(model.state_dict(), "./original.pth")

model.load_state_dict(torch.load('./original.pth'))

text = "The Farmer and the Stork"
print("===========Greedy====================",file=open("one.txt", "a"))
print(predict(model, my_data.vocab, text, 1000),file=open("one.txt", "a"))
print("===========Sampling==================",file=open("one.txt", "a"))
print(predict(model, my_data.vocab, text, 1000, sampling=True),file=open("one.txt", "a"))

text = "The Elephant and the Wagoner"
print("===========Greedy====================",file=open("two.txt", "a"))
print(predict(model, my_data.vocab, text, 1000),file=open("two.txt", "a"))
print("===========Sampling==================",file=open("two.txt", "a"))
print(predict(model, my_data.vocab, text, 1000, sampling=True),file=open("two.txt", "a"))

text = "The Student and the pan"
print("===========Greedy====================",file=open("three.txt", "a"))
print(predict(model, my_data.vocab, text, 1000),file=open("three.txt", "a"))
print("===========Sampling==================",file=open("three.txt", "a"))
print(predict(model, my_data.vocab, text, 1000, sampling=True),file=open("three.txt", "a"))

text = "A bread is never eaten"
print("===========Greedy====================",file=open("four.txt", "a"))
print(predict(model, my_data.vocab, text, 1000),file=open("four.txt", "a"))
print("===========Sampling==================",file=open("four.txt", "a"))
print(predict(model, my_data.vocab, text, 1000, sampling=True),file=open("four.txt", "a"))