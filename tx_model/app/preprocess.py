import os
from io import open
import torch
from torch.utils.data import Dataset
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from google.cloud import bigquery
import subprocess
import gzip
import pickle
from argparse import ArgumentParser
import multiprocessing as multi
import time
import math

class BigQuery(object):
    def __init__(self, isCached=False, isLocal=False):
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

        #if self.isLocal:
        #    query = query + '\nlimit 10000'
        query = query + '\nlimit 10000'

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

    def detach_spark(self):
        self.spark.stop()
        del self.spark
        pass


class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []
        self.token_embeddings = []

        self.user2idx = {}
        self.idx2user = []

        self.cat2idx = {}
        self.idx2cat = []

    def add_token(self, token, emb=None):
        if token is None:
            print('None token')
        if token not in self.token2idx:
            self.idx2token.append(token)
            if emb is None:
                emb = np.random.rand(512).astype('float32')
            self.token_embeddings.append(emb)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def add_user(self, user_id):
        if user_id not in self.user2idx:
            self.idx2user.append(user_id)
            self.user2idx[user_id] = len(self.idx2user) - 1
        return self.user2idx[user_id]

    def add_category(self, cat):
        if cat not in self.cat2idx:
            self.idx2cat.append(cat)
            self.cat2idx[cat] = len(self.idx2cat) - 1
        return self.cat2idx[cat]


class Features(object):
    def __init__(self, dataset_name, seq_len, feature_set, isCached=False, isLocal=False):
        self.bq = BigQuery(isCached, isLocal)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.feature_set = feature_set
        self.dictionary = Dictionary()
        self.dictionary.add_token('<cls>')
        self.dictionary.add_token('<pad>')
        self.dictionary.add_token('<unk>')

        # Load pretrained embeddings
        # First two colums are index and merchant_name
        if dataset_name=='merchant_seqs_by_tx':
            emb_data = self.bq.load_embeddings()

            # Add words to the dictionary
            for emb in emb_data:
                token = emb[1]
                embedding = emb[2:].astype(np.float32)
                self.dictionary.add_token(token, embedding)
            print('Merchant vocab size: {0}'.format(len(self.dictionary.idx2token)))

        raw_train = self.bq.load_sequences(self.dataset_name, self.seq_len, 'train')
        raw_val = self.bq.load_sequences(self.dataset_name, self.seq_len, 'val')
        raw_test = self.bq.load_sequences(self.dataset_name, self.seq_len, 'test')

        del self.bq

        processing_start = time.time()

        self.train_data = self.prepare_data(raw_train, 'train')
        self.val_data = self.prepare_data(raw_val, 'val')
        self.test_data = self.prepare_data(raw_test, 'test')

        processing_end = time.time()
        processing_duration = round(processing_end - processing_start, 1)
        print('Total processing time: {0}'.format(processing_duration))

        self.ntoken = len(self.dictionary.idx2token)
        self.nusers = len(self.dictionary.idx2user)
        self.ncat = len(self.dictionary.idx2cat)

        # Upload to GCS
        upload_cmd = 'gsutil -m cp -r datasets/ gs://tensorboard_logging/'
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
        if len(input) < self.max_len-1:
            input = np.append(input, np.repeat('<pad>', self.max_len-len(input)))

        return input, target


    def prepare_numeric_sequence(self, seq):
        sequence = np.append(0, np.array(seq))
        target = sequence[-1]
        input = sequence[:-1]

        # Adding padding
        if len(input) < self.max_len-1:
            input = np.append(input, np.repeat(0, self.max_len-len(input)))

        return input, target


    def tensor(self, X):
        #print('pre: {}'.format(type(X)))
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
        #print('post: {}'.format(type(X)))
        return X


    def encode_cyclical(self, X):
        X = X.type(torch.FloatTensor)
        sin = torch.sin(X)
        cos = torch.cos(X)
        return torch.squeeze(torch.stack((sin, cos),1))


    def amount_scaler(self, X):
        # Assumes min = 0 and max = 3000
        Xsc = X / 3000
        return torch.unsqueeze(Xsc, dim=-1)


    def process_row(self, chunk):
        start_time = time.time()
        input_dict = {}
        target_dict = {}

        log_interval = 100
        samples = []
        for index, row in enumerate(chunk):
            auxilliary_features = []
            for feature, config in self.feature_set.items():
                if config['enabled']:
                    if feature =='user_reference':
                        input_dict[feature] = self.tensor(
                                self.dictionary.add_user(
                                        row[self.schema_dict[feature]]
                                        )
                                )
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))

                    if feature == 'merchant_name':
                        sequence, target = self.prepare_token_sequence(
                                row[self.schema_dict[feature]]
                                )
                        token_ids = np.array([], dtype=np.int64)
                        mask = np.array([])
                        for token in sequence:
                            token_ids = np.append(token_ids, [self.dictionary.add_token(token)])
                            if token != '<pad>':
                                mask = np.append(mask, [1])
                            else:
                                mask = np.append(mask, [0])
                        input_dict[feature] = self.tensor(token_ids)
                        target_dict[feature] = self.tensor(self.dictionary.add_token(target))
                        #masks.append(self.tensor(mask))
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))
                        #print('{0} target: {1}'.format(feature, target_dict[feature]))

                    if feature == 'sys_category':
                        sequence, target = self.prepare_token_sequence(
                                row[self.schema_dict[feature]]
                                )
                        token_ids = np.array([], dtype=np.int64)
                        for token in sequence:
                            token_ids = np.append(token_ids, [self.dictionary.add_category(token)])
                        input_dict[feature] = self.tensor(token_ids)
                        target_dict[feature] = self.tensor(self.dictionary.add_category(target))
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))
                        #print('{0} target: {1}'.format(feature, target_dict[feature]))

                    if feature in ['day_of_week', 'eighth_of_day']:
                        sequence, target = self.prepare_numeric_sequence(
                                row[self.schema_dict[feature]]
                                )
                        cyc = self.encode_cyclical(self.tensor(sequence))
                        auxilliary_features.append(cyc)
                        target_dict[feature] = self.tensor(target)
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))
                        #print('{0} target: {1}'.format(feature, target_dict[feature]))

                    if feature == 'amount':
                        sequence, target = self.prepare_numeric_sequence(
                                row[self.schema_dict[feature]]
                                )
                        scaled = self.amount_scaler(self.tensor(sequence))
                        auxilliary_features.append(scaled)
                        target_dict[feature] = self.amount_scaler(self.tensor(target))


            if len(auxilliary_features) > 0:
                auxilliary_features = torch.squeeze(torch.cat(auxilliary_features, 1))
                input_dict['aux'] = auxilliary_features

            sample = input_dict, target_dict
            samples.append(sample)

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
        schema_list = [key for key in self.schema_dict]

        seq_lengths = [len(seq) for seq in data[:,self.schema_dict['merchant_name']]]
        self.max_len = max(seq_lengths)
        self.min_len = min(seq_lengths)
        print('Max seq_len: {}'.format(self.max_len))
        print('Min seq_len: {}'.format(self.min_len))

        cores = multi.cpu_count()
        chunk_size = math.ceil(len(data) / cores)
        print('Running {0} chunks of {1} rows'.format(cores, chunk_size))
        log_interval = round(chunk_size / 100)

        chunks = []
        for pnum in range(cores):
            start_id = pnum * chunk_size
            stop_id = start_id + chunk_size
            chunks.append(data[start_id:stop_id])

        with multi.Pool(processes=cores) as pool:
            self.samples = pool.map(self.process_row, chunks)
            pool.close()
            pool.join()

        print(len(self.samples))
        self.samples = [item for sublist in self.samples for item in sublist]
        print(len(self.samples))

        # Write to disk
        dir = os.path.join('datasets', self.dataset_name + '_' + str(self.seq_len))
        try:
            os.mkdir('datasets')
            os.mkdir(dir)
        except FileExistsError:
            pass
        path = os.path.join(dir, split)
        print('Writing {0} to {1}'.format(split, path))
        with open(path, 'wb') as f:
            pickle.dump(self.samples, f)

        return self.samples


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data', type=str, default='merchant_seqs_by_tx',
                        help='Name of input data')
    parser.add_argument('--seq_len', type=int, default=32,
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

    Features(args.data, args.seq_len, feature_set)
