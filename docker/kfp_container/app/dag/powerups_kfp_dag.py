import datetime

from airflow import models
from airflow.operators.bash_operator import BashOperator

default_args = {
    'depends_on_past': False,
    'start_date': datetime.datetime(year=2019, month=1, day=7, hour=21),
    'email': ['shweta@koho.ca'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with models.DAG(
        'kf-pipeline-dag',
        schedule_interval=None,
        catchup=False,
        template_searchpath='/home/airflow/gcs/data/kubeflow_pipeline/',
        default_args=default_args) as dag:

    run_copy_files = BashOperator(
        task_id='copy-required-files',
        bash_command='gsutil -m cp -r gs://powerups_personalization/kubeflow_pipeline /home/airflow/gcs/data/'
    )

    run_kf_create_cluster = BashOperator(
        task_id='create-kubernetes-cluster',
        bash_command='gcloud container clusters create kf-pipeline --zone us-central1-a --machine-type=n1-standard-4 '
    )

    run_kfp_get_credential = BashOperator(
        task_id='get-kubernetes-credential',
        bash_command='gcloud container clusters get-credentials kf-pipeline --zone us-central1-a --project tensile-oarlock-191715'
    )

    run_kf_deploy_kfp = BashOperator(
        task_id='deploy-kubeflow',
        bash_command='kubectl apply -f /home/airflow/gcs/data/kubeflow_pipeline/namespaced-install.yaml '
    )

    # to wait until the deployments are complete.
    run_check_deployment_success = BashOperator(
        task_id='check_deployment_success',
        bash_command='kubectl rollout status -w -n kubeflow deployment.v1.apps/proxy-agent && kubectl rollout status -w -n kubeflow deployment.v1.apps/ml-pipeline ',
        retries=3
    )

    run_pipeline = BashOperator(
        task_id='run-powerups-personalisation-training-pipeline',
        bash_command='python /home/airflow/gcs/data/kubeflow_pipeline/powerups_kf_pipeline.py'
    )

    # To get the UI URL:
    # kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com


    # run_delete_cluster = BashOperator(
    #     task_id='delete-kubernetes-cluster',
    #     bash_command='gcloud container clusters delete kf-test --zone us-central1-a -q'
    # )

    run_kfp_get_credential.set_upstream(run_kf_create_cluster)
    run_copy_files.set_upstream(run_kfp_get_credential)
    run_kf_deploy_kfp.set_upstream(run_copy_files)
    run_check_deployment_success.set_upstream(run_kf_deploy_kfp)
    run_pipeline.set_upstream(run_check_deployment_success)
    #  run_delete_cluster.set_upstream(run_pipeline)
