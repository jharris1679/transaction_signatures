from model import TransactionSignatures
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import subprocess
import pickle
import data
import os
import re


ckpt_dir = 'checkpoints/tx_model_v1/'
min_loss = 100
for ckpt in os.listdir(ckpt_dir):
    loss = float(re.search(r'(?<=val_loss=)\d+.\d+', ckpt).group(0))
    version_name = re.search(r'(?<=tx_model_v1_)(.*)(?=-epoch)', ckpt).group(0)
    if loss < min_loss:
        ckpt_path = os.path.join(ckpt_dir, ckpt)
        print('Loading {0}'.format(ckpt_path))
        min_loss = loss


model = TransactionSignatures.load_from_checkpoint(ckpt_path)

idx2merch = pd.DataFrame(model.dataset.dictionary['idx2merchant'], columns=['merchant_name'])
proj_2D = pd.DataFrame(model.merchant_embedding.weight.detach().numpy())
print(trained_embeddings.head())
assert len(idx2merch)==trained_embeddings.shape[0]

embeddings_table = idx2merch.join(trained_embeddings)
print(embeddings_table.head())

filename = version_name + '_merchant_embedding.csv'
filepath = os.path.join('embeddings', filename)
embeddings_table.to_csv(filepath, index=False)

copy_string = 'gsutil -m -q cp -r {0} gs://tx_sig_embeddings/{1}'
copy_command = copy_string.format(filepath, filename)
subprocess.run(copy_command.split())
