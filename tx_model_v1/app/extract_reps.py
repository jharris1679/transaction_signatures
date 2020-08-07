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
idx2user = pd.DataFrame(model.dataset.dictionary['idx2user'], columns=['user_reference'])

merchant_embeddings = pd.DataFrame(model.merchant_embedding.weight.detach().numpy())
user_embeddings = pd.DataFrame(model.user_embedding.weight.detach().numpy())

print(merchant_embeddings.head())
assert len(idx2merch)==merchant_embeddings.shape[0]

print(user_embeddings.head())
assert len(idx2user)==user_embeddings.shape[0]

merchant_table = idx2merch.join(merchant_embeddings)
print(merchant_table.head())

user_table = idx2user.join(user_embeddings)
print(user_table.head())

filename = version_name + '_merchant_embedding.csv'
filepath = os.path.join('embeddings', filename)
merchant_table.to_csv(filepath, index=True, header=False)
copy_string = 'gsutil -m -q cp -r {0} gs://tx_sig_embeddings/{1}'
copy_command = copy_string.format(filepath, filename)
subprocess.run(copy_command.split())

filename = version_name + '_user_embedding.csv'
filepath = os.path.join('embeddings', filename)
user_table.to_csv(filepath, index=True, header=False)
copy_string = 'gsutil -m -q cp -r {0} gs://tx_sig_embeddings/{1}'
copy_command = copy_string.format(filepath, filename)
subprocess.run(copy_command.split())
