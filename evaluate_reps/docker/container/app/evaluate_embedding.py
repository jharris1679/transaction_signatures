import pandas as pd
import pandas_gbq as pgbq
import numpy as np
from scipy import stats
import joblib
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import datetime
import glob
from google.cloud import bigquery
import subprocess
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


bq_client = bigquery.Client()

def execute_query(experiment_name):
    # Set the destination table
    job_config = bigquery.QueryJobConfig()
    dest_table_ref = bq_client.dataset('merchant_vectors').table(experiment_name)
    job_config.destination = dest_table_ref
    job_config.write_disposition = 'WRITE_TRUNCATE'

    # open query file
    with open('query_lib/'+experiment_name+'.sql') as f:
        query = f.read()

    print('Running {0} query'.format(experiment_name))

    # Start the query, passing in the extra configuration.
    query_job = bq_client.query(
        query,
        location="US",
        job_config=job_config,
        project='koho-staging'
    )

    query_job.result()  # Waits for the query to finish
    print("Query results loaded to table {}".format(dest_table_ref.path))

def load_and_aggregate(experiment_name):
    print('Starting Spark')
    #SparkContext.setSystemProperty('spark.executor.memory', '15g')
    #sc = SparkContext("local", "eval_merchant_embedding")
    spark = SparkSession.builder \
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.13.1-beta") \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "25g") \
        .config("spark.driver.maxResultSize", "5g") \
        .config("spark.dynamicAllocation.enabled ", True) \
        .appName("eval_merchant_embedding") \
        .getOrCreate()

    print(spark.sparkContext.getConf().getAll())

    # Use the Cloud Storage bucket for temporary BigQuery export data used
    # by the connector.
    bucket = 'gs://merchant-embeddings/'+experiment_name+'/tmp'
    spark.conf.set('temporaryGcsBucket', bucket)

    # Read the data from BigQuery as a Spark Dataframe.
    spark_df = spark.read.format('bigquery') \
        .option('table', 'merchant-embeddings:merchant_vectors.'+experiment_name) \
        .load()
    print(len(spark_df.columns))

    print('Aggregating')
    spark_agg = spark_df.groupBy('created_at', 'user_reference', 'merchant_type').mean()
    print(len(spark_agg.columns))
    spark_df = None

    print('Input vectors length: {0}'.format(spark_agg.count()))

    return spark_agg


# Execute the script
start = datetime.datetime.now()
print(start)

experiment_name = sys.argv[1]
print(experiment_name)

# Pull new inference data and save as a table in BQ
execute_query(experiment_name)

# Use pyspark to load prepare inputs
input_vectors = load_and_aggregate(experiment_name)
print(len(input_vectors.columns))

seed = 7

# Define holdout set as transactions made on or after Feb 1 2020
train = np.array(input_vectors.where(input_vectors.created_at < '2020-02-01').collect())
val = np.array(input_vectors.where(input_vectors.created_at >= '2020-02-01').collect())
input_vectors = None

# Define inputs and outputs
X_train = train[:,3:]
X_val = val[:,3:]
y_train = train[:,2]
y_val = val[:,2]

print(X_train[:5])
print(y_train[:5])

# Train model
xgb_model = xgb.XGBClassifier(n_estimators=1000,
                              max_depth=5,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective='multi:softprob',
                              nthread=-1,
                              random_state=seed)

xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_val)

print(metrics.accuracy_score(y_val, y_pred))
print(metrics.confusion_matrix(y_val, y_pred))
print(metrics.classification_report(y_val, y_pred))

duration = datetime.datetime.now() - start
print(datetime.datetime.now())
print(duration)

with open('saved_models/'+experiment_name+'.save', 'wb') as f:
    joblib.dump(xgb_model, f)
