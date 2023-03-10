import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.text_dataset import ChunkedTextData

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