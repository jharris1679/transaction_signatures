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
model.prepare_data()
#trainer = pl.Trainer()
#trainer.test(model)
predictions = model.predict(model.test_data[:10])

for i, pred in enumerate(predictions):
    print('pred: {}'.format(pred))
    sample = model.test_data[i]
    target = sample[1]
    target_merchant_ids = target['merchant_name']
    target_merchant_names = []
    for id in target_merchant_ids:
        target_merchant_names.append(model.dataset.dictionary['idx2merchant'][id])
    print('target: {}'.format(target_merchant_names))
