---
title: Home
layout: home
nav_order: 1
---

## A generative model to mitigate bias in tabular data.

As the use of AI and other machine learning models becomes more prevalent in our day-to-day lives, it becomes increasingly important to scrutinize the underlying datasets that these models are trained on so that we avoid perpetuating any potential biases.

Machine learning(ML) models are not limited to just entertainment; they can be used to make serious life decisions, such as predicting recidivism, medical-delivery priorities, mortgage loans, or career progressions. At the end of the day, biased data leads to biased models.

This project aims to solve the problem and started as a UC Berkeley capstone, to generate a synthetic training data that will mitigate racial, gender or other forms of bias in datasets. We attempted to solve the problem by Generative Advesarial Network(GAN) transfomer architecture with three discirminator, one classifier and one generator.

### Installing

Pull down model from GitHub and follow virtual environment set up instructions below. 

[Visit Project GitHub Repo](https://github.com/tflint-ucb/fair_transformer_GAN){: .btn .btn-purple }

### Docker container (set up option 1 of 3)
1. Clone Git repo (fair_transformer_GAN)
2. Build the docker image
```
docker build -t <image name> -f fair_transformer_GAN/setup/Dockerfile .
```
3. Run jupyter notebook
```
docker run -p 8888:8888 <docker image name>
```
4. Copy the http jupyter notebook link into browser
5. Can continue running the following "Getting Started" code to train model and generate data. Can also browse to the example python notebook from the Git repo. 
6. After generating the mitigated bias data, don't forget to save your local machine.
```
docker cp my_container:/path/to/*.npy /path/to/local/dir
```
or download via jupyter notebook. 

7. Once stop your container, don't forget to remove the image and container if no longer in use. 
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


### Local (set up option 2 of 3)
1. Clone Git repo
2. Create a pip directory
```
mkdir ~/.pip/
```
3. Create an empty pip.conf file in that directory
```
~/.pip/pip.conf 
```
4. Run the setup_env.sh script
```
source setup/setup_env.sh
```
### Google Colab (set up option 3 of 3)
todo: link to a old distro of code that allowed for transgan called from google colab --- there's a link that can point to a google colab 

### Getting started

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

Read your data into pandas
```
df = pd.read_csv('data/raw/adult.csv')
df.head()
```

Create a Dataset object, which will return a numpy array that's ready for the model. This will do a couple basic [pre-processing steps] and checks to get the data ready for the model
```
dataset = Dataset(df)
# saves processed data to interim/processed folder
np_input = dataset.pre_process(protected_var='race', 
                                outcome_var='income', 
                                # specify path here
                                output_file_name='data/interim/adult_race_multi', multiclass=True)
np_input
```

If you would like to uplevel your data, please specify the numpy input of your data (if not using the output from our preprocessor), the percentage you would like to uplevel, and the outcome class you would like to balance. Output pickle file created. Upleveling is a 
technique of modifying a dataset to increase the representation of a particular group or subgroup.
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

Initialize the model 
```
input_data_file='data/interim/adult_race_multi.pkl' 
output_file='output/adult_race_fair_trans_gan/'
```
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

Generate data in the following way

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
Calculate metrics
```
orig_data = np.load(input_data_file, allow_pickle = True)
orig_data.shape
```

```
output_gen_X = np.load('data/generated/adult_race_fair_trans_gan_GEN/.npy')
output_gen_Y = np.load('data/generated/adult_race_fair_trans_gan_GEN/_y.npy')
output_gen_z = np.load('data/generated/adult_race_fair_trans_gan_GEN/_z.npy')

output_gen = np.c_[output_gen_z, output_gen_X, output_gen_Y]

output_gen
```

```
# original shape and gen data shape same
orig_data = orig_data[:-42,]
print(output_gen.shape == orig_data.shape)
```

```
#convert to df
gen_df = pd.DataFrame(output_gen)
orig_df = pd.DataFrame(orig_data)
```

```
# metrics evaluating the generated data
metrics = Metrics()
metrics.multi_fair_data_generation_metrics(gen_df, orig_df)
```
```
# train a classifier using our script (or train your own) and return outputs required to evaluate the classification metrics
TestX, TestY, TestPred = Classifier.logistic_regression(gen_df, orig_df)
```
```
# metrics evaluating the classifier trained on the generated data and predicted on the original data
metrics.multi_fair_classification_metrics(TestX, TestY, TestPred)
```

[pre-processing steps]: #
