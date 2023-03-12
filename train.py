import os
import torch
from src.text_dataset import LongTextData
from src.model import LSTMModel
from src.utils import train
from config.defaults import get_cfg_defaults

def main():
    cfg = get_cfg_defaults()
    cwd = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_data = LongTextData(cwd + cfg.TEXT_PATH, device=device)
    
    vocab_length = len(my_data.vocab)

    model = LSTMModel(vocab_length, embedding_length=cfg.BPTT_LEN, n_hidden=cfg.N_HIDDEN, n_layers=cfg.N_LAYERS)

    train(model, my_data=my_data, epochs = cfg.NUM_EPOCHS, batch_size = cfg.BATCH_SIZE, seq_length=cfg.BPTT_LEN, lr = cfg.LR, clip = 1.0, device=device)
    torch.save(model.state_dict(), cwd + f"/results/model_{cfg.LR}_{cfg.BPTT_LEN}.pth")

if __name__ == '__main__':
    main()