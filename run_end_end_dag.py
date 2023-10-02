from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'khizar',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'run_docker_script_on_ec2',
    default_args=default_args,
    description='Run script in Docker container on EC2',
    schedule_interval='*/10 * * * *',  # Every 10 minutes
    catchup=False,  # Skip missed runs
)

start = DummyOperator(
    task_id='start',
    dag=dag,
)

end = DummyOperator(
    task_id='end',
    dag=dag,
)

run_preprocessing_script = SSHOperator(
    ssh_conn_id='ssh_default',
    task_id='run_preprocess_script',
    command='docker exec end-end-container-name python3 -m preprocessing.preprocess_yelp_review_dataset --dataset_path my_datasets/yelp_dataset/yelp_academic_dataset_review.csv',
    dag=dag,
)

run_training_script = SSHOperator(
    ssh_conn_id='ssh_default',
    task_id='run_preprocess_script',
    command='docker exec end-end-container-name python3 -m train',
    dag=dag,
)


start >> run_preprocessing_script >> run_training_script >> end