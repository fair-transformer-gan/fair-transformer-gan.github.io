---
title: Home
layout: home
nav_order: 1
---

# A Generative Model to Mitigate Bias in data

As AI and machine learning models become more ubiquitous in our daily lives, it is crucial that we scrutinize the datasets used to train them to avoid perpetuating biases. This is particularly important since Machine Learning (ML) models are increasingly being used to make critical decisions that impact people's lives, such as predicting recidivism, medical prioritization, mortgage approvals, and career advancement.

To tackle this issue, our team at UC Berkeley embarked on a Capstone Project with the goal of generating synthetic training data that would minimize biases linked to ptorected attributes. Protected attribute are qualities or characteristics that by law, cannot be discriminated against (Ex: race, gender, nationality, etc.)

We built off the existing FairGAN+ model by applying a transformer architecture and multi-class protected attribute support (max 5 classes). The objective was to develop a model that would produce less biased data and, as a result, create fairer outcomes.

## Setup

Pull the GitHub repo and follow one of the setup steps. After properly setting up your virtual environment, you can follow the [Getting Started](#getting-started) code to train the model and generate data. You can also browse to the example Python Jupyter Notebook in the GitHub repo.  

[Visit Project GitHub Repo](https://github.com/tflint-ucb/fair_transformer_GAN){: .btn .btn-purple }


### Option 1: Local/Cloud Development

**Note:** This setup does not work on Mac laptop with M1 processor. This setup works on other Linux-based machines, including Mac w/ Intel processor, AWS EC2, AWS SageMaker, AWS SageMaker Studio Lab, GCP, etc.

1. Clone Git repo (fair_transformer_GAN)
2. Create a pip directory
```
mkdir ~/.pip/
```
3. Create an empty pip.conf file in that directory
```
~/.pip/pip.conf 
```
4. Run the setup_env.sh script from repo's root directory 
```
source setup/setup_env.sh
```

### Option 2: Docker Container 

1. Clone Git repo (fair_transformer_GAN)
2. Build the docker image
```
docker build -t <image name> -f fair_transformer_GAN/setup/Dockerfile .
```
3. Run the Jupyter Notebook
```
docker run -p 8888:8888 <docker image name>
```
4. Copy the http jupyter notebook link into browser
5. Continue running the following [Getting Started](#getting-started) code to train model and generate data. You can also browse to the example python notebook from the Git repo. 

Note:
After generating the data and/or models, save your data to your local machine. You can also download your data to your local machine via the Jupyter Notebook terminal.
```
docker cp my_container:/path/to/*.npy /path/to/local/dir
```

Helpful docker commands:
```
# see what containers are active or stopped (exited)
docker ps -a 
# stop container
docker stop <container id>
docker ps -a 
# remove container
docker rm <container id>
docker ps -a
docker images
# rm images 
docker rmi <images id>
docker images
```





## Getting Started

Import necessary dependencies

```
import numpy as np
import pandas as pd
import tensorflow as tf
from src.dataset.dataset import Dataset
from src.model.fair_transformer_gan import FairTransformerGAN
from src.metrics.metrics import Metrics
from src.metrics.classifier import Classifier
```

You can read in your raw data into a pandas dataframe and take advantage of built in pre-processing steps. 

```
df = pd.read_csv('data/raw/adult.csv')
df.head()
```

Create a Dataset object and pre-process the data. This includes one-hot encoding categorical columns, scaling numeric columns, checking for nulls, etc. The pre-processing function also saves the data to the output file specified. Make sure you have already created all the subfolders in the file path.
```
dataset = Dataset(df)
# saves processed data to interim/processed folder
np_input = dataset.pre_process(protected_var='race', 
                                outcome_var='income', 
                                output_file_name='data/interim/adult_race_multi', multiclass=True)
np_input
```

{: .note }
The steps above are **Optional** feel free to pre-process your own data and save a pickled numpy array for the model. See [Dataset](/docs/dataset.html) class API for more details.

Upleveling is the technique of modifying a dataset to increase the representation of a particular group or subgroup. If you would like to uplevel your data, please specify the data in numpy format, the percentage you would like to uplevel, and the outcome class you would like to balance. Output pickle file created. 
```
np_upleveled = dataset.uplevel(np_data = np_input, percentage = 2, balanceOutcome = 1,
                                    output_file_name_path='data/interim/adult_race_multi_upleveled')
```

Get the distribution of protected attribute and the outcome variable
```
# get distribution of protected attribute race
p_z = dataset.get_protected_distribution()
p_z
```
```
# get distribution of outcome variable
p_y = dataset.get_target_distribution()
p_y
```

Specify the path to the input numpy data and the path to save the model. Make sure you have already created all the subfolders in the file paths.
```
input_data_file='data/interim/adult_race_multi.pkl' 
output_file='output/adult_race_fair_trans_gan/'
```
Initialize the model 
```
fairTransGAN = FairTransformerGAN(dataType='count',
                                    inputDim=np_input.shape[1] - 2,
                                    embeddingDim=128,
                                    randomDim=128,
                                    generatorDims=(128, 128),
                                    discriminatorDims=(256, 128, 1),
                                    compressDims=(),
                                    decompressDims=(),
                                    bnDecay=0.99,
                                    l2scale= 0.001,
                                    lambda_fair=1)
```

Train model
```
# clear any tf variables in current graph
tf.reset_default_graph()
```
```
fairTransGAN.train(dataPath=input_data_file,
                    modelPath="",
                    outPath=output_file,
                    pretrainEpochs=1,
                    nEpochs=1,
                    generatorTrainPeriod=1,
                    discriminatorTrainPeriod=1,
                    pretrainBatchSize=100,
                    batchSize=100,
                    saveMaxKeep=0,
                    p_z = p_z,
                    p_y = p_y)
```

Generate less-biased data. Specify the path to the trained model file and the path to save the generated data. Make sure you have already created all the subfolders in the file paths.

```
# clear any tf variables in current graph
tf.reset_default_graph()
```

```
#  generate synthetic data using the trained model 
fairTransGAN.generateData(nSamples=np_input.shape[0],
                modelFile='output/adult_race_fair_trans_gan/a1_1681280765-0',
                batchSize=100,
                outFile='data/generated/adult_race_fair_trans_gan_GEN/',
                p_z = p_z,
                p_y = p_y)
```

Load in the orginal data 
```
orig_data = np.load(input_data_file, allow_pickle = True)
orig_data.shape
```
Concatenate the generated z protected attribute data, x data, y outcome data together
```
output_gen_X = np.load('data/generated/adult_race_fair_trans_gan_GEN/.npy')
output_gen_Y = np.load('data/generated/adult_race_fair_trans_gan_GEN/_y.npy')
output_gen_z = np.load('data/generated/adult_race_fair_trans_gan_GEN/_z.npy')

output_gen = np.c_[output_gen_z, output_gen_X, output_gen_Y]

output_gen
```

```
# resize original data to be the same shape as generated data
orig_data = orig_data[:-42,]
print(output_gen.shape == orig_data.shape)
```

```
# convert numpy objects to df
gen_df = pd.DataFrame(output_gen)
orig_df = pd.DataFrame(orig_data)
```

Calculate fairness metrics on generated data
```
# metrics evaluating the generated data
metrics = Metrics()
metrics.multi_fair_data_generation_metrics(gen_df)
```
```
# train a classifier using our logistic regression model (or use your own classifier) and return classification metrics
TestX, TestY, TestPred = Classifier.logistic_regression(gen_df, orig_df)
```
```
# metrics evaluating the classifier trained on the generated data and predicted on the original data
metrics.multi_fair_classification_metrics(TestX, TestY, TestPred)
```
```
# train a classifier using our random forest model (or use your own classifier) and return classification metrics
TestX_r, TestY_r, TestPred_r = Classifier.random_forest(gen_df, orig_df)
```
```
# metrics evaluating the classifier trained on the generated data and predicted on the original data
metrics.multi_fair_classification_metrics(TestX_r, TestY_r, TestPred_r)
```
```
# calculate euclidean distance metric
metrics.euclidean_distance(gen_df, orig_df)
```
