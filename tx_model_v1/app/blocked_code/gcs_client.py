from google.cloud import storage
import pytorch_lightning as pl
import torch
import gzip
import os
import re

class LogSyncCallback(pl.Callback):
    def __init__(self, log_interval, model_name, experiment_id):

        self.log_interval = log_interval
        self.model_name = model_name
        self.experiment_id = experiment_id

        client  = storage.Client()
        self.logs_bucket = client.get_bucket('tensorboard_logging')
        self.ckpt_bucket = client.get_bucket('tx_sig_checkpoints')

        # For finding best ckpt
        self.min_ckpt_loss = 100


    def on_validation_end(self, trainer, pl_module):
        self.sync_logs()
        self.sync_ckpt()
        pass


    def on_batch_start(self, trainer, pl_module):
        log_cadence = trainer.batch_idx%self.log_interval
        if log_cadence==0 and log_cadence!=0:
            self.sync_logs()
        pass


    def sync_logs(self):
        log_path = os.path.join('lightning_logs/', self.model_name, self.experiment_id)
        print('\nSyncing {} to GCS'.format(log_path))

        for file in os.listdir(log_path):
            filepath = os.path.join(log_path, file)
            log_blob = self.logs_bucket.blob(filepath)
            log_blob.upload_from_filename(filepath)

        pass


    def sync_ckpt(self):
        ckpt_path = os.path.join('checkpoints', self.model_name)

        for filename in os.listdir(ckpt_path):
            loss = int(re.search(r'(?<=val_loss=)\d', filename).group(0))

            if loss < self.min_ckpt_loss:
                ckpt_file = os.path.join(ckpt_path, filename)
                print('\nSyncing {} to GCS'.format(ckpt_file))
                ckpt_blob = self.ckpt_bucket.blob(os.path.join(self.model_name, filename))
                ckpt_blob.upload_from_filename(ckpt_file)

                self.min_ckpt_loss = loss

        pass
