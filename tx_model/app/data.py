import os
import numpy as np
import subprocess
import gzip
import pickle

class LoadDataset(object):
    def __init__(self,
                 gcs_source='gs://tx_sig_datasets/merchant_seqs_by_tx_32_data/',
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

        self.gcs_source = gcs_source
        self.download_destination = download_destination

        if local_source is None:
            # Set local_source with dataset name
            self.local_source = self.download_destination + gcs_source.split('/')[3]
            self.download_data(gcs_source, download_destination)

        for filename in os.listdir(self.local_source):
            print(filename)
            path = os.path.join(self.local_source, filename)
            with open(path, 'rb') as f:
                file = pickle.load(f)
                setattr(self, filename, file)

        self.nmerchant = len(self.dictionary['idx2merchant'])
        self.nusers = len(self.dictionary['idx2user'])
        self.ncat = len(self.dictionary['idx2cat'])

    def download_data(self, gcs_source, download_destination):
        mkdir_cmd = 'mkdir -p {0}'.format(download_destination)
        if not os.path.exists(download_destination):
            print('Running {0}'.format(mkdir_cmd))
            subprocess.run(mkdir_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        dload_cmd = 'gsutil -m cp -r {0} {1}'.format(gcs_source, download_destination)
        print('Running {0}'.format(dload_cmd))
        result = subprocess.run(dload_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            return result.stdout
        else:
            if result.stderr:
                print(result.stderr)
