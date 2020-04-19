import os
import numpy as np
import subprocess
import gzip
import pickle

class GCSDataset(object):
    def __init__(self,
                 gcs_path='gs://tensorboard_logging/datasets/merchant_seqs_by_tx_32/',
                 local_data_root='gcs_data/'):
        """
        gcs_path is the gs://dir containing data

        Files in gcs_path should include:
         - train
         - val
         - test
         - dictionary
        """
        self.gcs_path = gcs_path
        self.local_data_root = local_data_root
        self.local_path = self.local_data_root + gcs_path.split('/')[4]

        self.download_data(gcs_path, local_data_root)

        for filename in os.listdir(self.local_path):
            path = os.path.join(self.local_path, filename)
            with open(path, 'rb') as f:
                setattr(self, filename, pickle.load(f))

        self.ntoken = len(self.dictionary['idx2token'])
        self.nusers = len(self.dictionary['idx2user'])
        self.ncat = len(self.dictionary['idx2cat'])

    def download_data(self, gcs_path, local_data_root):
        mkdir_cmd = 'mkdir -p {0}'.format(local_data_root)
        if not os.path.exists(local_data_root):
            print('Running {0}'.format(mkdir_cmd))
            subprocess.run(mkdir_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        dload_cmd = 'gsutil -m cp -r {0} {1}'.format(gcs_path, local_data_root)
        print('Running {0}'.format(dload_cmd))
        result = subprocess.run(dload_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            return result.stdout
        else:
            if result.stderr:
                raise result.stderr
