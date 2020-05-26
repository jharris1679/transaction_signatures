import os
import torch
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from google.cloud import bigquery
import subprocess
import pickle
from argparse import ArgumentParser
import multiprocessing as multi
import time
import math
import json
import shutil

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

    def execute_query(self, query_name):
        # Set the destination table
        job_config = bigquery.QueryJobConfig()
        self.dest_table_ref = self.bq_client.dataset(self.bq_dataset).table(query_name)
        job_config.destination = self.dest_table_ref
        job_config.write_disposition = 'WRITE_TRUNCATE'

        # open query file
        with open('queries/{0}.sql'.format(query_name)) as f:
            query = f.read()

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

    def load_sequences(self, dataset_name, split_type):

        query_name = '{0}_{1}'.format(dataset_name, split_type)

        if not self.isCached:
            job_complete, job_exception = self.execute_query(query_name)
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
        self.mcc2idx = {}

        self.idx2merchant = []
        self.idx2user = []
        self.idx2mcc = [k for k in self.mcc2idx]


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


    def add_mcc(self, mcc):
        if mcc not in self.mcc2idx:
            self.idx2mcc.append(mcc)
            self.mcc2idx[mcc] = (len(self.idx2mcc)-1)
        return self.mcc2idx[mcc]


class Features(object):
    def __init__(self, args, isCached=False, isLocal=False):
        self.args = args

        # Initialize token dictionaries
        self.dictionary = Dictionary()
        self.dictionary.add_merchant('<cls>')
        self.dictionary.add_mcc('<cls>')
        self.dictionary.add_merchant('<pad>')
        self.dictionary.add_mcc('<pad>')

        # Initialize bigquery client with spark
        bq = BigQuery(args.sample_size, isCached, isLocal)

        # Open config file
        with open('feature_config.json', 'r') as config_file:
            self.feature_config = json.load(config_file)
        assert self.feature_config['merchant_name']['enabled'] == True

        # Use args to enable features
        self.feature_config['user_reference']['enabled'] = self.args.include_user_context
        self.feature_config['eighth_of_day']['enabled'] = self.args.include_eighth_of_day
        self.feature_config['day_of_week']['enabled'] = self.args.include_day_of_week
        self.feature_config['amount']['enabled'] = self.args.include_amount
        self.feature_config['mcc']['enabled'] = self.args.include_mcc

        # Write updated config back to disk
        with open('feature_config.json', 'w') as config_file:
            json.dump(self.feature_config, config_file)

        # Create directories
        if args.sample_size==-1:
            sample_dir = 'sample_all'
        else:
            sample_dir = 'sample_{0}'.format(str(args.sample_size))
        dataset_dir  = '{0}_{1}_data'.format(self.args.dataset_name, str(self.args.seq_len))
        self.data_dir = os.path.join(sample_dir, dataset_dir)
        try:
            os.mkdir(sample_dir)
            os.mkdir(self.data_dir)
        except FileExistsError:
            pass

        # Load pretrained embeddings
        # First two colums are index and merchant_name
        if args.use_pretrained_embeddings:
            emb_data = bq.load_embeddings()

            # Add words to the dictionary
            for emb in emb_data:
                merchant = emb[1]
                embedding = emb[2:].astype(np.float32)
                self.dictionary.add_merchant(merchant, embedding)
            print('Merchant vocab size: {0}'.format(len(self.dictionary.idx2merchant)))

        # Load data and execute processing logic
        processing_start = time.time()

        raw_train = bq.load_sequences(self.args.dataset_name, 'train')
        self.prepare_data(raw_train, 'train')
        raw_train = None

        raw_val = bq.load_sequences(self.args.dataset_name, 'val')
        self.prepare_data(raw_val, 'val')
        raw_val = None

        raw_test = bq.load_sequences(self.args.dataset_name, 'test')
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


    def stdscaler_fit(self, X):
        self.s = np.std(X)
        self.u = np.mean(X)
        pass


    def stdscaler_transform(self, X):
        Xsc = ( X - self.u ) / self.s
        X = None
        return Xsc


    def prepare_data(self, data, split):
        try:
            os.mkdir('tmp_data')
        except FileExistsError:
            pass

        # A structure to enable indexing a numpy matrix with column names.
        user_col = self.feature_config['user_reference']['column_index']
        merchant_col = self.feature_config['merchant_name']['column_index']
        dow_col = self.feature_config['day_of_week']['column_index']
        eod_col = self.feature_config['eighth_of_day']['column_index']
        amount_col = self.feature_config['amount']['column_index']
        mcc_col = self.feature_config['mcc']['column_index']

        # Scale amount values
        scaling_start = time.time()

        if self.feature_config['amount']['enabled']==True:
            if split=='train':
                amount_seqs = data[:,amount_col]
                amounts = [math.ceil(x/10)*10 for seq in amount_seqs for x in seq]
                amount_seqs = None
                self.stdscaler_fit(amounts)
                amounts = None

            for idx, seq in enumerate(data[:,amount_col]):
                seq_sc = self.stdscaler_transform(seq)
                data[idx, amount_col] = seq_sc

        scaling_end = time.time()
        scaling_duration = round(scaling_end - scaling_start, 1)
        print('scaling time: {0}s'.format(scaling_duration))

        # tokenize pre-concurrency
        tokenization_start = time.time()

        self.data_length = 0
        for row in data:
            self.dictionary.add_user(row[user_col])
            self.data_length += math.ceil(len(row[merchant_col])/self.args.seq_len)
            for merchant in row[merchant_col]:
                self.dictionary.add_merchant(merchant)
            for mcc in row[mcc_col]:
                self.dictionary.add_mcc(str(mcc))

        tokenization_end = time.time()
        tokenization_duration = round(tokenization_end - tokenization_start, 1)
        print('tokenization time: {0}s'.format(tokenization_duration))

        # write to be read by row.transform
        path = os.path.join('tmp_data', 'dictionary')
        with open(path, 'wb') as f:
            pickle.dump(self.dictionary.__dict__, f)

        # Divide data into chunks
        cores = multi.cpu_count()
        num_chunks = cores*2
        chunk_size = math.ceil(len(data) / num_chunks)
        print('Running {0} chunks of {1} rows'.format(num_chunks, chunk_size))
        chunks = []
        for pnum in range(num_chunks):
            start_id = pnum * chunk_size
            stop_id = start_id + chunk_size
            chunks.append((pnum, data[start_id:stop_id]))

        # Free up RAM
        data = None

        # Process data concurrently, one process per cpu core.
        # Not guaranteed to use all cores.
        row = Row(self.args)
        with multi.Pool(processes=cores) as pool:
            processed_chunk_files = pool.map(row.transform, chunks)
            pool.close()
            pool.join()
        print('Mulitprocessing complete')

        # Free up RAM
        chunks = None

        # Write to disk
        merge_start = time.time()

        out_path = os.path.join(self.data_dir, split)
        print('Writing {0} to {1}'.format(split, out_path))
        data = []
        with open(out_path, 'wb') as outfile:
            pickle.dump(self.data_length, outfile)
            for chunk_file in processed_chunk_files:
                print('Merging {0}'.format(chunk_file))
                source_path = os.path.join('tmp_data', chunk_file)
                with open(source_path, 'rb') as readfile:
                    chunk_data = pickle.load(readfile)
                    data.extend(chunk_data)
            # Delete tmp_data
            shutil.rmtree('tmp_data')
            pickle.dump(data, outfile)
        data = None

        merge_end = time.time()
        merge_duration = round(merge_end - merge_start, 1)
        print('merge time: {0}s'.format(merge_duration))



        return None


class Row(object):
    def __init__(self, args):
        self.args = args

        path = os.path.join('tmp_data', 'dictionary')
        with open(path, 'rb') as dict:
            self.dictionary = pickle.load(dict)
            del self.dictionary['merchant_embeddings']

        with open('feature_config.json', 'r') as config_file:
            self.feature_config = json.load(config_file)

    def split_sequences(self, seqs):
        seqs = np.column_stack(seqs)
        #print('presplit: {}'.format(seqs.shape))
        subseqs = []
        for i in range(0, len(seqs), self.args.seq_len):
            subseq = seqs[i:i+self.args.seq_len]
            #print('subseq: {}'.format(len(subseq)))
            subseqs.append(subseq)
        return subseqs


    def prepare_token_subseq(self, subseq, feat2idx):
        sequence = np.append('<cls>', np.array(subseq)[::-1])
        input = sequence[:-1]
        target = sequence[1:]
        assert len(target)==len(input)

        # Adding padding
        if len(input) < self.args.seq_len:
            input = np.append(input, np.repeat('<pad>', self.args.seq_len-len(input)))
            target = np.append(target, np.repeat('<pad>', self.args.seq_len-len(target)))
        assert_string = "Mismatched seqlens: {0} {1} {2}\n{3}".format(len(target), len(input), self.args.seq_len, target)
        assert len(target)==len(input)==self.args.seq_len, assert_string

        input_ids = []
        target_ids = []
        for index, input_value in enumerate(input):
            input_ids.append(feat2idx[input_value])
            target_ids.append(feat2idx[target[index]])

        return input_ids, target_ids


    def prepare_numeric_subseq(self, subseq):
        sequence = np.append(0, np.array(subseq)[::-1]).astype(np.float64)
        input = sequence[:-1]
        target = sequence[1:]
        assert len(target)==len(input)

        # Adding padding
        if len(input) < self.args.seq_len:
            input = np.append(input, np.repeat(0, self.args.seq_len-len(input)))
            target = np.append(target, np.repeat(0, self.args.seq_len-len(target)))
        assert_string = "Mismatched seqlens: {0} {1} {2}\n{3}".format(len(target), len(input), self.args.seq_len, target)
        assert len(target)==len(input)==self.args.seq_len, assert_string

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

    # This fucntion executes once per process
    def transform(self, chunk):
        idx = chunk[0]
        chunk = chunk[1]
        #print('chunk len: {}'.format(len(chunk)))

        start_time = time.time()

        log_interval = 10000
        samples = []
        enabled_features = {k:v for k,v in self.feature_config.items() if v['enabled']==True}
        for index, user_row in enumerate(chunk):
            #print('user len: {}'.format(len(user_row[1])))
            # user in col 0 is not a sequence, don't split
            subseqs = self.split_sequences(user_row[1:])
            #print('num subseqs: {}'.format(len(subseqs)))
            for index, subseq in enumerate(subseqs):
                #print('subseq shape: {}'.format(subseq.shape))
                input_dict = {}
                target_dict = {}
                for feature, config in enabled_features.items():
                    col_idx = self.feature_config[feature]['column_index']
                    # we've separated col 0 and so col_idx-1
                    feat_seq = subseq[:,col_idx-1]
                    #print('feat_seq: {}'.format(feat_seq.shape))

                    if feature =='user_reference':
                        user = user_row[col_idx]
                        input_dict[feature] = self.tensor(self.dictionary['user2idx'][user])

                    if feature == 'merchant_name':
                        feat2idx = self.dictionary['merchant2idx']
                        input_ids, target_ids = self.prepare_token_subseq(feat_seq, feat2idx)
                        padding_mask = []
                        for merchant in enumerate(input_ids):
                            if merchant == '<pad>':
                                padding_mask.append(1)
                            else:
                                padding_mask.append(0)
                        input_dict[feature] = self.tensor(np.array(input_ids))
                        target_dict[feature] = self.tensor(np.array(target_ids))
                        padding_mask = torch.tensor(padding_mask).bool()

                    if feature == 'mcc':
                        feat2idx = self.dictionary['mcc2idx']
                        input_ids, target_ids = self.prepare_token_subseq(feat_seq, feat2idx)
                        input_dict[feature] = self.tensor(np.array(input_ids))
                        target_dict[feature] = self.tensor(np.array(target_ids))

                    if feature in ['day_of_week', 'eighth_of_day']:
                        inputs, targets = self.prepare_numeric_subseq(feat_seq)
                        input_dict[feature] = self.encode_cyclical(self.tensor(inputs))
                        target_dict[feature] = self.tensor(targets)

                    if feature == 'amount':
                        inputs, targets = self.prepare_numeric_subseq(feat_seq)
                        input_dict[feature] = torch.unsqueeze(self.tensor(inputs), dim=-1)
                        target_dict[feature] = torch.unsqueeze(self.tensor(targets), dim=-1)

                sample = input_dict, target_dict, padding_mask
                samples.append(sample)

            if index%log_interval==0 and index!=0:
                stop_time = time.time()
                avg_duration = round(stop_time - start_time, 1) / log_interval
                print('{0} rows complete\n{1}s/row'.format(index, avg_duration))
                start_time = time.time()

        # test to ensure chunk isn't homogeneous
        if idx==0:
            distinct_targets = set()
            for row in samples:
                distinct_targets.add(row[1]['merchant_name'])
            assert len(distinct_targets) > 1

        #print('samples length: {}'.format(len(samples)))
        chunk_filename = 'chunk_{0}'.format(idx)
        path = os.path.join('tmp_data', chunk_filename)
        with open(path, 'wb') as fh:
            pickle.dump(samples, fh)
        return chunk_filename




if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='user_power',
                        help='Name of input data')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='sequence length')
    parser.add_argument('--sample_size', type=int, default=-1,
                        help='sequence length')
    parser.add_argument('--use_pretrained_embeddings', action='store_true',
                        help='Use pretrained embeddings or not')
    parser.add_argument('--include_user_context', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_eighth_of_day', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_day_of_week', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_amount', action='store_true',
                        help='Turn on feature')
    parser.add_argument('--include_mcc', action='store_true',
                        help='Turn on feature')

    args = parser.parse_args()

    print(args)

    Features(args)
