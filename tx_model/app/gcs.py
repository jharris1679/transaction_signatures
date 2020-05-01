from google.cloud import storage
import pytorch_lightning as pl
import subprocess
import torch
import gzip
import os
import re

class LogSyncCallback(pl.Callback):
    def __init__(self, log_interval, model_name, experiment_id):

        self.log_interval = log_interval
        self.model_name = model_name
        self.experiment_id = experiment_id

        # For finding best ckpt
        self.min_ckpt_loss = 100


    def on_batch_start(self, trainer, pl_module):
        log_cadence = trainer.batch_idx%self.log_interval
        if log_cadence==0 and trainer.batch_idx!=0:
            self.sync_logs()
        pass


    def on_epoch_start(self, trainer, pl_module):
        self.sync_logs()
        self.sync_ckpt()
        pass


    def on_train_end(self, trainer, pl_module):
        self.sync_logs()
        self.sync_ckpt()
        pass


    def sync_logs(self):
        log_path = os.path.join('lightning_logs/', self.model_name, self.experiment_id)
        print('\nSyncing {} to GCS'.format(log_path))
        copy_string = 'gsutil -m -q cp -r {0} gs://tensorboard_logging/lightning_logs/{1}/'
        copy_command = copy_string.format(log_path, self.model_name)
        subprocess.run(copy_command.split())
        pass


    def sync_ckpt(self):
        ckpt_path = os.path.join('checkpoints', self.model_name)

        for filename in os.listdir(ckpt_path):
            loss = int(re.search(r'(?<=val_loss=)\d', filename).group(0))

            if loss < self.min_ckpt_loss:
                ckpt_file = os.path.join(ckpt_path, filename)
                print('\nSyncing {} to GCS'.format(ckpt_file))
                copy_string = 'gsutil -m cp -r {0} gs://tx_sig_checkpoints/{1}/'
                copy_command = copy_string.format(ckpt_file, self.model_name)
                subprocess.run(copy_command.split())

                self.min_ckpt_loss = loss
        pass
