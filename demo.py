import os
import torch
from src.text_dataset import LongTextData
from src.model import LSTMModel
from src.utils import predict
from config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()
cwd = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_data = LongTextData(cwd + cfg.TEXT_PATH, device=device)

vocab_length = len(my_data.vocab)

model = LSTMModel(vocab_length, embedding_length=cfg.BPTT_LEN, n_hidden=cfg.N_HIDDEN, n_layers=cfg.N_LAYERS)

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