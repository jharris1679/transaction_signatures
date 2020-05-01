import os
import torch
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from google.cloud import bigquery
import subprocess
import pickle
from argparse import ArgumentParser
import multiprocessing as multi
import time
import math
import json

class BigQuery(object):
    def __init__(self, sample_size, isCached=False, isLocal=False):
        self.sample_size = sample_size
        self.isCached = isCached
        self.isLocal = isLocal
        print(self.isLocal)
        print('Starting Spark')
        self.spark = SparkSession.builder \
            .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.11:0.13.1-beta") \
            .config("spark.executor.memory", "6g") \
            .config("spark.driver.memory", "25g") \
            .config("spark.driver.maxResultSize", "5g") \
            .config("spark.dynamicAllocation.enabled ", True) \
            .appName("embed_users") \
            .getOrCreate()

        # Use the Cloud Storage bucket for temporary BigQuery export data used
        # by the connector.
        self.bucket = 'gs://merchant-embeddings/tmp'
        self.spark.conf.set('temporaryGcsBucket', self.bucket)

        # Configure bigquery
        print('Initializing BigQuery Client')
        self.bq_dataset = 'user_vectors'
        self.bq_client = bigquery.Client()

    def execute_query(self, query_name, seq_len=None):
        # Set the destination table
        job_config = bigquery.QueryJobConfig()
        self.dest_table_ref = self.bq_client.dataset(self.bq_dataset).table(query_name)
        job_config.destination = self.dest_table_ref
        job_config.write_disposition = 'WRITE_TRUNCATE'

        # open query file
        with open('queries/{0}.sql'.format(query_name)) as f:
            query = f.read()

        if seq_len is not None:
            query = query.replace('{{seq_len}}', str(seq_len))

        if self.sample_size > 0:
            query = query + '\nlimit {0}'.format(self.sample_size)

        # Start the query, passing in the extra configuration.
        print('Running {0} query'.format(query_name))
        query_job = self.bq_client.query(
            query,
            location="US",
            job_config=job_config,
            project='tensile-oarlock-191715'
        )

        # Waits for the query to finish
        job_complete = False
        while job_complete==False:
            job_exception = query_job.exception()
            if job_exception:
                print(job_exception)
                break
            job_complete = query_job.done()

        return job_complete, job_exception

    def load_sequences(self, dataset_name, seq_len, split_type):

        query_name = '{0}_{1}'.format(dataset_name, split_type)

        if not self.isCached:
            job_complete, job_exception = self.execute_query(query_name, seq_len)
        else:
            job_complete = True

        # Read the data from BigQuery as a Spark Dataframe.
        if job_complete and not job_exception:
            print("Query results loaded to table {}".format(self.dest_table_ref.path))
            spark_df = self.spark.read.format('bigquery') \
                .option('table', 'merchant-embeddings:{0}.{1}'.format(self.bq_dataset, query_name)) \
                .load()
        else:
            print(job_exception)

        print('Number of sequences: {0}'.format(spark_df.count()))

        data = np.array(spark_df.collect())
        spark_df.unpersist()

        return data

    def load_embeddings(self):

        query_name = 'indexed_training_embeddings'

        if not self.isCached:
            self.execute_query(query_name)

        # Read the data from BigQuery as a Spark Dataframe.
        spark_df = self.spark.read.format('bigquery') \
            .option('table', 'merchant-embeddings:{0}.{1}'.format(self.bq_dataset, query_name)) \
            .load()

        print('Number of embeddings: {0}'.format(spark_df.count()))

        data = np.array(spark_df.collect())
        spark_df.unpersist()

        return data


class Dictionary(object):
    def __init__(self):
        self.merchant_embeddings = []

        self.merchant2idx = {}
        self.user2idx = {}
        with open('cat2idx.json', 'r') as f:
            self.cat2idx = json.load(f)

        self.idx2merchant = []
        self.idx2user = []
        self.idx2cat = [k for k in self.cat2idx]


    def add_merchant(self, merchant, emb=None):
        if merchant is None:
            print('None merchant')
        if merchant not in self.merchant2idx:
            self.idx2merchant.append(merchant)
            if emb is None:
                emb = np.random.rand(512).astype('float32')
            self.merchant_embeddings.append(emb)
            self.merchant2idx[merchant] = (len(self.idx2merchant)-1)
        return self.merchant2idx[merchant]


    def add_user(self, user_id):
        if user_id not in self.user2idx:
            self.idx2user.append(user_id)
            self.user2idx[user_id] = (len(self.idx2user)-1)
        return self.user2idx[user_id]


class Features(object):
    def __init__(self,
                 dataset_name,
                 seq_len,
                 sample_size,
                 feature_set,
                 isCached=False,
                 isLocal=False):
        bq = BigQuery(sample_size, isCached, isLocal)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.feature_set = feature_set
        self.dictionary = Dictionary()
        self.dictionary.add_merchant('<cls>')
        self.dictionary.add_merchant('<pad>')
        self.dictionary.add_merchant('<unk>')

        if sample_size==-1:
            sample_dir = 'sample_all'
        else:
            sample_dir = 'sample_{0}'.format(str(sample_size))
        dataset_dir  = '{0}_{1}_data'.format(self.dataset_name, str(self.seq_len))
        self.data_dir = os.path.join(sample_dir, dataset_dir)
        try:
            os.mkdir(sample_dir)
            os.mkdir(self.data_dir)
        except FileExistsError:
            pass

        # Load pretrained embeddings
        # First two colums are index and merchant_name
        if dataset_name=='merchant_seqs_by_tx':
            emb_data = bq.load_embeddings()

            # Add words to the dictionary
            for emb in emb_data:
                merchant = emb[1]
                embedding = emb[2:].astype(np.float32)
                self.dictionary.add_merchant(merchant, embedding)
            print('Merchant vocab size: {0}'.format(len(self.dictionary.idx2merchant)))

        processing_start = time.time()

        raw_train = bq.load_sequences(self.dataset_name, self.seq_len, 'train')
        self.prepare_data(raw_train, 'train')
        raw_train = None

        raw_val = bq.load_sequences(self.dataset_name, self.seq_len, 'val')
        self.prepare_data(raw_val, 'val')
        raw_val = None

        raw_test = bq.load_sequences(self.dataset_name, self.seq_len, 'test')
        self.prepare_data(raw_test, 'test')
        raw_test = None

        processing_end = time.time()
        processing_duration = round(processing_end - processing_start, 1)
        print('Processing time: {0}s'.format(processing_duration))

        # Write dictionary to disk
        path = os.path.join(self.data_dir, 'dictionary')
        print('Writing {0} to {1}'.format('dictionary', path))
        with open(path, 'wb') as f:
            pickle.dump(self.dictionary.__dict__, f)

        # Upload to GCS
        cmd_string = 'gsutil -m cp -r {0}/ gs://tx_sig_datasets/{1}/'
        upload_cmd = cmd_string.format(self.data_dir, sample_dir)
        print('Running {0}'.format(upload_cmd))
        result = subprocess.run(upload_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            pass
        else:
            if result.stderr:
                raise result.stderr


    def prepare_token_sequence(self, seq):
        sequence = np.append('<cls>', np.array(seq))
        target = sequence[-1]
        input = sequence[:-1]

        # Adding padding
        if len(input) < self.seq_len:
            input = np.append(input, np.repeat('<pad>', self.seq_len-len(input)))

        return input, target


    def prepare_numeric_sequence(self, seq):
        sequence = np.append(0, np.array(seq))
        target = sequence[-1]
        input = sequence[:-1]

        # Adding padding
        if len(input) < self.seq_len:
            input = np.append(input, np.repeat(0, self.seq_len-len(input)))

        return input, target


    def tensor(self, X):
        if type(X)==np.ndarray:
            if X.dtype==np.float64:
                X = torch.tensor(X).type(torch.FloatTensor)
            else:
                X = torch.tensor(X).type(torch.int64)
        elif type(X)==np.int64:
            X = torch.tensor(X).type(torch.int64)
        elif type(X)==np.float64:
            X = torch.tensor(X).type(torch.FloatTensor)
        else:
            X = torch.tensor(X).type(torch.int64)
        return X


    def encode_cyclical(self, X):
        X = X.type(torch.FloatTensor)
        sin = torch.sin(X)
        cos = torch.cos(X)
        return torch.squeeze(torch.stack((sin, cos),1))


    def stdscaler_fit(self, X):
        self.s = round(np.std(X), 3)
        self.u = round(np.mean(X), 3)
        pass


    def stdscaler_transform(self, X):
        Xsc = round(( X - self.u ) / self.s, 3)
        X = None
        return Xsc


    def process_row(self, chunk):
        # This fucntion executes once per process
        start_time = time.time()
        input_dict = {}
        target_dict = {}

        log_interval = 10000
        samples = []
        for index, row in enumerate(chunk):
            enabled_features = {k:v for k,v in self.feature_set.items() if v['enabled']==True}

            for feature, config in enabled_features.items():
                feat_seq = row[self.schema_dict[feature]]

                if feature =='user_reference':
                    user = feat_seq
                    input_dict[feature] = self.tensor(self.dictionary.user2idx[user])

                if feature == 'merchant_name':
                    sequence, target = self.prepare_token_sequence(feat_seq)
                    merchant_ids = []
                    masks = []
                    for merchant in sequence:
                        merchant_ids.append(self.dictionary.merchant2idx[merchant])
                        if merchant != '<pad>':
                            masks.append(1)
                        else:
                            masks.append(0)
                    input_dict[feature] = self.tensor(np.array(merchant_ids))
                    target_dict[feature] = self.tensor(self.dictionary.merchant2idx[target])

                if feature == 'sys_category':
                    sequence, target = self.prepare_token_sequence(feat_seq)
                    cat_ids = []
                    for cat in sequence:
                        cat_ids.append(self.dictionary.cat2idx[cat])
                    input_dict[feature] = self.tensor(cat_ids)
                    target_dict[feature] = self.tensor(self.dictionary.cat2idx[target])

                if feature in ['day_of_week', 'eighth_of_day']:
                    sequence, target = self.prepare_numeric_sequence(feat_seq)
                    cyc = self.encode_cyclical(self.tensor(sequence))
                    input_dict[feature] = cyc
                    target_dict[feature] = self.tensor(target)

                if feature == 'amount':
                    sequence, target = self.prepare_numeric_sequence(feat_seq)
                    sequence = torch.unsqueeze(self.tensor(sequence), dim=-1)
                    input_dict[feature] = sequence
                    target_dict[feature] = self.tensor(target)

            chunk = None
            sample = input_dict, target_dict
            input_dict = None
            target_dict = None
            samples.append(sample)
            sample = None

            if index%log_interval==0:
                stop_time = time.time()
                avg_duration = round(stop_time - start_time, 1) / log_interval
                print('{0} rows complete\n{1}s/row'.format(index, avg_duration))
                start_time = time.time()

        return samples


    def prepare_data(self, data, split):
        # A structure to enable indexing a numpy matrix with column names.
        self.schema_dict = {'auth_ts': 0,
                    'user_reference': 1,
                    'merchant_name': 2,
                    'day_of_week': 3,
                    'eighth_of_day': 4,
                    'amount': 5,
                    'sys_category': 6,
                    'auth_ts_seq': 7}

        # Scale amount values
        scaling_start = time.time()

        if self.feature_set['amount']['enabled']==True:
            if split=='train':
                amount_seqs = data[:,self.schema_dict['amount']]
                amounts = [x for seq in amount_seqs for x in seq]
                amount_seqs = None
                self.stdscaler_fit(amounts)
                amounts = None

            for idx, seq in enumerate(data[:,self.schema_dict['amount']]):
                seq_sc = self.stdscaler_transform(seq)
                data[idx, self.schema_dict['amount']] = seq_sc

        scaling_end = time.time()
        scaling_duration = round(scaling_end - scaling_start, 1)
        print('scaling time: {0}s'.format(scaling_duration))

        # tokenize pre-concurrency
        tokenization_start = time.time()

        for row in data:
            self.dictionary.add_user(row[self.schema_dict['user_reference']])
            for merchant in row[self.schema_dict['merchant_name']]:
                self.dictionary.add_merchant(merchant)

        tokenization_end = time.time()
        tokenization_duration = round(tokenization_end - tokenization_start, 1)
        print('tokenization time: {0}s'.format(tokenization_duration))

        # Divide data into one chunk per cpu core
        cores = multi.cpu_count()
        chunk_size = math.ceil(len(data) / cores)
        print('Running {0} chunks of {1} rows'.format(cores, chunk_size))
        chunks = []
        for pnum in range(cores):
            start_id = pnum * chunk_size
            stop_id = start_id + chunk_size
            chunks.append(data[start_id:stop_id])

        # Free up RAM
        data = None

        # Process data concurrently, one process per cpu core.
        # Not guaranteed to use all cores.
        with multi.Pool(processes=cores) as pool:
            processed_chucks = pool.map(self.process_row, chunks)
            pool.close()
            pool.join()

        # Free up RAM
        chunks = None

        # Flatten list of mutliprocess outputs
        samples = [item for sublist in processed_chucks for item in sublist]

        # Free up RAM
        processed_chucks = None

        # Write to disk
        path = os.path.join(self.data_dir, split)
        print('Writing {0} to {1}'.format(split, path))
        with open(path, 'wb') as f:
            pickle.dump(samples, f)

        # Free up RAM
        samples = None

        return None


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data', type=str, default='merchant_seqs_by_tx',
                        help='Name of input data')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='sequence length')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='sequence length')
    parser.add_argument('--include_user_context', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_eighth_of_day', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_day_of_week', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_amount', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_sys_category', action='store_true',
                        help='Turn on feature')

    args = parser.parse_args()

    print(args)

    feature_set = {'merchant_name':
                        {'enabled': True},
                    'user_reference':
                        {'enabled': args.include_user_context},
                    'eighth_of_day':
                        {'enabled': args.include_eighth_of_day},
                    'day_of_week':
                        {'enabled': args.include_day_of_week},
                    'amount':
                        {'enabled': args.include_amount},
                    'sys_category':
                        {'enabled': args.include_sys_category}
                    }

    Features(args.data, args.seq_len, args.sample_size, feature_set)
