from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os


# The DAG object; we'll need thid to instantiate a DAG
from airflow import DAG
training_pipeline = None
# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

# [END imporETL DAG tutorial_prediction',
# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
with DAG(
        'ocr',
        default_args={'retries': 2},
        # [END default_args]
        description='Captcha recognition',
        schedule_interval="@weekly",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        catchup=False,
        tags=['example'],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # [START extract_function]

    from ocr.pipeline.training_pipeline import TrainPipeline

    training_pipeline = TrainPipeline()


    def data_ingestion(**kwargs):
        ti = kwargs['ti']
        data_ingestion_artifacts = training_pipeline.start_data_ingestion()
        print(data_ingestion_artifacts)
        ti.xcom_push('data_ingestion_artifact', data_ingestion_artifacts.to_dict())

    def data_transformation(**kwargs):
        from ocr.entity.artifact_entity import DataIngestionArtifacts
        ti = kwargs['ti']
        data_ingestion_artifacts = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifacts")
        data_ingestion_artifacts = DataIngestionArtifacts(*(data_ingestion_artifacts))
        data_transformation_artifact = training_pipeline.start_data_transformation(
            data_ingestion_artifacts=data_ingestion_artifacts
        )
        ti.xcom_push('data_transformation_artifact', data_transformation_artifact.to_dict())


    def model_trainer(**kwargs):
        from ocr.entity.artifact_entity import DataTransformationArtifacts
        ti = kwargs['ti']
        data_transformation_artifacts = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifacts")
        data_transformation_artifacts = DataTransformationArtifacts(*(data_transformation_artifacts))

        model_trainer_artifacts = training_pipeline.start_model_trainer(
            data_transformation_artifacts=data_transformation_artifacts)
        ti.xcom_push('model_trainer_artifact', model_trainer_artifacts.to_dict())

    def push_model(**kwargs):
        ti = kwargs['ti']
        model_trainer_artifacts = ti.xcom_pull(task_ids="model_trainer", key="model_trainer_artifacts")

        model_pusher_artifacts = training_pipeline.start_model_pusher()
        print(f'Model pusher artifacts: {model_pusher_artifacts}')
        ti.xcom_push('model_trainer_artifacts', model_trainer_artifacts.to_dict())

        print("Training pipeline completed")


    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
    )
    data_ingestion.doc_md = dedent(
        """\
    #### Extract task
    A simple Extract task to get data ready for the rest of the data pipeline.
    In this case, getting data is simulated by reading from a hardcoded JSON string.
    This data is then put into xcom, so that it can be processed by the next task.
    """
    )
    data_transformation = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation
    )

    model_trainer = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer

    )
    push_model = PythonOperator(
        task_id="push_model",
        python_callable=push_model

    )

    data_ingestion >> data_transformation >> model_trainer >> push_model
