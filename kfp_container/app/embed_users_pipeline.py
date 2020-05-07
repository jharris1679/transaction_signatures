import kfp
from kfp import dsl
import os
import random
import csv

experiment_id = os.getenv('EXP_ID')

def get_experiment_id():
    with open('/home/helper/words.csv', 'r') as f:
        words = csv.reader(f)
        wordlist = list(words)
    random.shuffle(wordlist)
    experiment_id = '{0}_{1}'.format(wordlist[0][0], wordlist[1][0])
    print('Experiment ID: {0}'.format(experiment_id))
    return experiment_id

def training_op():
    return dsl.ContainerOp(
        name='train-data',
        image='gcr.io/tensile-oarlock-191715/user-embedding-img:{0}'.format(experiment_id)
    )

@dsl.pipeline(
    name='User Embedding',
    description='Spending Signatures'
)

def embed_user_pipeline():
    #data = dataset_op()
    training = training_op()
    training.set_memory_request('90G')
    training.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-v100')
    training.add_resource_limit('nvidia.com/gpu', 2)
    #training.after(data)

if __name__ == '__main__':
    pipeline_func = embed_user_pipeline
    pipeline_filename = pipeline_func.__name__ + '.yaml'
    kfp.compiler.Compiler().compile(pipeline_func, pipeline_filename)
    client = kfp.Client()
    experiment = client.create_experiment("User_Embeddings")
    if len(experiment_id) > 0:
        run_name = experiment_id
    else:
        run_name = get_experiment_id()
    print('Submitting run: {0}'.format(run_name))
    run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename)
