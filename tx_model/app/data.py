from torch.utils.data import Dataset
import numpy as np
import subprocess
import gzip
import pickle
import time
import os

class Dataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as fh:
            self.len = pickle.load(fh)
            self.data = pickle.load(fh)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx]

class LoadDataset(object):
    def __init__(self,
                 sample_size,
                 gcs_source='gs://tx_sig_datasets/sample_{0}/merchant_seqs_by_tx_power_32_data/',
                 download_destination='gcs_data/',
                 local_source=None):
        """
        gcs_source: The gs://dir containing data
        download_destination: Where data will be downloaded to
        local_source: Don't download, load files from this directory

        Downloads data from GCS by default. To load files locally,
        set local_source with directory containing data files.

        Files in source directory must be:
         - train
         - val
         - test
         - dictionary
        """
        self.download_destination = download_destination

        if sample_size < 0:
            self.gcs_source = gcs_source.format('all')
        else:
            self.gcs_source = gcs_source.format(sample_size)

        if local_source is None:
            # Set local_source with dataset name
            self.local_source = self.download_destination + gcs_source.split('/')[4]
        else:
            self.local_source = local_source
            print('Reading data from local files')


    def download_dict(self):
        mkdir_cmd = 'mkdir -p {0}'.format(self.download_destination)
        if not os.path.exists(self.download_destination):
            print('Running {0}'.format(mkdir_cmd))
            subprocess.run(mkdir_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        src_path = self.gcs_source + "dictionary"
        dload_cmd = 'gsutil -m cp -r {0} {1}'.format(src_path, self.download_destination)
        print('Running {0}'.format(dload_cmd))
        result = subprocess.run(dload_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            pass
        else:
            if result.stderr:
                print(result.stderr)
                return result.stderr

        path  = os.path.join(self.local_source, 'dictionary')
        with open(path, 'rb') as f:
            dict = pickle.load(f)
            setattr(self, 'dictionary', dict)

        self.nmerchant = len(self.dictionary['idx2merchant'])
        self.nusers = len(self.dictionary['idx2user'])
        self.ncat = len(self.dictionary['idx2cat'])


    def download_data(self):
        mkdir_cmd = 'mkdir -p {0}'.format(self.download_destination)
        if not os.path.exists(self.download_destination):
            print('Running {0}'.format(mkdir_cmd))
            subprocess.run(mkdir_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        dload_cmd = 'gsutil -m cp -r {0} {1}'.format(self.gcs_source, self.download_destination)
        print('Running {0}'.format(dload_cmd))
        result = subprocess.run(dload_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            pass
        else:
            if result.stderr:
                print(result.stderr)
                return result.stderr

        read_start = time.time()

        for filename in os.listdir(self.local_source):
            path = os.path.join(self.local_source, filename)
            print('Reading {0}'.format(path))
            if filename != 'dictionary':
                with open(path, 'rb') as f:
                    data = Dataset(path)
                    setattr(self, filename, data)

        read_end = time.time()
        read_duration = round(read_end - read_start, 1)
        print('read time: {0}s'.format(read_duration))
