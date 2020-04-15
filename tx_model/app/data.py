import os
from io import open
import torch
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from google.cloud import bigquery

class BigQuery(object):
    def __init__(self, isCached=False, isLocal=False):
        self.isCached = isCached
        self.isLocal = isLocal
        print(self.isLocal)
        print('Starting Spark')
        self.spark = SparkSession.builder \
            .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.11:0.13.1-beta") \
            .config("spark.executor.memory", "25g") \
            .config("spark.driver.memory", "25g") \
            .config("spark.driver.maxResultSize", "5g") \
            .appName('embed_users') \
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

        if self.isLocal:
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

        #sorted_df = spark_df.sort(col('auth_ts').asc())
        #spark_df = None

        data = np.array(spark_df.collect())
        spark_df = None

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

        self.emb_data = np.array(spark_df.collect())
        spark_df = None

        return self.emb_data


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

        self.train_data = self.load('train')
        self.val_data = self.load('val')
        print(self.val_data[0])
        self.test_data = self.load('test')

        self.ntoken = len(self.dictionary.idx2token)
        self.nusers = len(self.dictionary.idx2user)
        self.ncat = len(self.dictionary.idx2cat)


    def prepare_token_sequence(self, seq):
        # query provides tx in descending temporal order, reverse to get
        # chronological sequence
        sequence = np.append('<cls>', np.array(seq)[::-1])
        target = sequence[-1]
        input = sequence[:-1]

        # Adding padding
        if len(input) < self.max_len:
            input = np.append(input, np.repeat('<pad>', self.max_len-len(input)))

        return input, target


    def prepare_numeric_sequence(self, seq):
        # query provides tx in descending temporal order, reverse to get
        # chronological sequence
        sequence = np.append(0, np.array(seq)[::-1])
        target = sequence[-1]
        input = sequence[:-1]

        # Adding padding
        if len(input) < self.max_len:
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

    def arrange_features(self, data):
        # A structure to enable indexing a numpy matrix with column names.
        schema_dict = {'auth_ts': 0,
                    'user_reference': 1,
                    'merchant_name': 2,
                    'day_of_week': 3,
                    'eighth_of_day': 4,
                    'amount': 5,
                    'sys_category': 6,
                    'auth_ts_seq': 7}
        schema_list = [key for key in schema_dict]

        seq_lengths = [len(seq) for seq in data[:,schema_dict['merchant_name']]]
        self.max_len = max(seq_lengths)
        self.min_len = min(seq_lengths)
        print('Max seq_len: {}'.format(self.max_len))
        print('Min seq_len: {}'.format(self.min_len))

        samples = []

        for row in data:
            input_dict = {}
            target_dict = {}

            for feature, config in self.feature_set.items():
                if config['enabled']:
                    if feature =='user_reference':
                        input_dict[feature] = self.tensor(self.dictionary.add_user(row[schema_dict[feature]]))
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))

                    if feature == 'merchant_name':
                        sequence, target = self.prepare_token_sequence(row[schema_dict[feature]])
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
                        sequence, target = self.prepare_token_sequence(row[schema_dict[feature]])
                        token_ids = np.array([], dtype=np.int64)
                        for token in sequence:
                            token_ids = np.append(token_ids, [self.dictionary.add_category(token)])
                        input_dict[feature] = self.tensor(token_ids)
                        target_dict[feature] = self.tensor(self.dictionary.add_category(target))
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))
                        #print('{0} target: {1}'.format(feature, target_dict[feature]))

                    if feature in ['day_of_week', 'eighth_of_day', 'amount']:
                        sequence, target = self.prepare_numeric_sequence(row[schema_dict[feature]])
                        input_dict[feature] = self.tensor(sequence)
                        target_dict[feature] = self.tensor(target)
                        #print('{0} input: {1}'.format(feature, input_dict[feature]))
                        #print('{0} target: {1}'.format(feature, target_dict[feature]))

            sample = (input_dict, target_dict)
            samples.append(sample)

        return samples


    def load(self, split):
        data = self.arrange_features(
                    self.bq.load_sequences(self.dataset_name, self.seq_len, split)
                )
        return data
