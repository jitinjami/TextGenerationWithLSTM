import torch.nn as nn

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