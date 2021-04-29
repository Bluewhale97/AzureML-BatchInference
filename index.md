## Introduction

Inferencing is used to perform new predictive tasks to the new observations using the trained model. We need to configure and deploy the trained model script, entry script, environment and so on. Sometimes we used to generate predictions from large numbers of observations in a batch process. To accomplish this, we can use Azure machine learning to publish a bacth inference pipeline.

In this article, we will talk about how to publish batch inference pipeline for a trained model as well as use a batch inference pipeline to generate predictions.

## 1. Creating a batch inference pipeline

First we still need to register a model: you can use the register method of the Model object

```python
from azureml.core import Model

classification_model = Model.register(workspace=your_workspace,
                                      model_name='classification_model',
                                      model_path='model.pkl', # local path
                                      description='A classification model')
 ```                                     
 
 Alternatively you also can use register_model method if you have a reference to the Run used to train the model.
 
 ```python
 run.register_model( model_name='classification_model',
                    model_path='outputs/model.pkl', # run outputs path
                    description='A classification model')
```

Now we require a scoring script to load the model and use it to predict new values. It must include two functions:

![image](https://user-images.githubusercontent.com/71245576/116602689-0e4ab700-a8fa-11eb-99a5-4c16a1eb55ac.png)

Typically, you use the init function to load the model from the model registry, and use the run function to generate predictions from each batch of data and return the results. The following example script shows this pattern:

```python
import os
import numpy as np
from azureml.core import Model
import joblib

def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)

def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList
```

Azure machine learning provides a type of pipeline step specifically for performing parallel batch inferencing. Using the ParallelRunStep class, you can read batches of files from a File dataset and write the processing output to a PipelineData reference. Additionally, you can set the output_action setting for the step to "append_row", which will ensure that all instances of the step being run in parallel will collate their results to a single output file named parallel_run_step.txt. The following code snippet shows an example of creating a pipeline with a ParallelRunStep:
```python
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Pipeline

# Get the batch dataset for input
batch_data_set = ws.datasets['batch-data']

# Set the output location
default_ds = ws.get_default_datastore()
output_dir = PipelineData(name='inferences',
                          datastore=default_ds,
                          output_path_on_compute='results')

# Define the parallel run step step configuration
parallel_run_config = ParallelRunConfig(
    source_directory='batch_scripts',
    entry_script="batch_scoring_script.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=aml_cluster,
    node_count=4)

# Create the parallel run step
parallelrun_step = ParallelRunStep(
    name='batch-score',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('batch_data')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)
# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
```

Finally we can run the pipeline and retrieve the step output:
```python
from azureml.core import Experiment

# Run the pipeline as an experiment
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Get the outputs from the first (and only) step
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='results')

# Find the parallel_run_step.txt file
for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root,file)

# Load and display the results
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]
print(df)
```
## 2. Publishing a batch inference pipeline

You can publish a batch inferencing pipeline as a REST service:

```python
published_pipeline = pipeline_run.publish_pipeline(name='Batch_Prediction_Pipeline',
                                                   description='Batch pipeline',
                                                   version='1.0')
rest_endpoint = published_pipeline.endpoint
```
Once published, you can use the service endpoint to initiate a batch inferencing job, as shown in the following example code:
```python
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "Batch_Prediction"})
run_id = response.json()["Id"]
```
You can also schedule the published pipeline to have it run automatically, as shown in the following example code:
```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

weekly = ScheduleRecurrence(frequency='Week', interval=1)
pipeline_schedule = Schedule.create(ws, name='Weekly Predictions',
                                        description='batch inferencing',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Batch_Prediction',
                                        recurrence=weekly)
```

## Reference

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/

