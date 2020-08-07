from model import TransactionSignatures
import pytorch_lightning as pl
import pickle
import data
import os
import re


ckpt_dir = 'checkpoints/tx_model_v1/'
min_loss = 100
for ckpt in os.listdir(ckpt_dir):
    loss = float(re.search(r'(?<=val_loss=)\d+.\d+', ckpt).group(0))
    if loss < min_loss:
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        print('Loading {0}'.format(ckpt_path))
        min_loss = loss


model = TransactionSignatures.load_from_checkpoint(ckpt_path)
trainer = pl.Trainer()
trainer.test(model)
