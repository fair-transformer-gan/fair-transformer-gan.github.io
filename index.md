---
title: Home
layout: home
---

## A generative model to mitigate bias in tabular data.

As the use of AI and other machine learning models becomes more prevalent in our day-to-day lives, it becomes increasingly important to scrutinize the underlying datasets that these models are trained on so that we avoid perpetuating any potential biases.

Machine learning(ML) models are not limited to just entertainment; they can be used to make serious life decisions, such as predicting recidivism, medical-delivery priorities, mortgage loans, or career progressions. At the end of the day, biased data leads to biased models.

This project aims to solve the problem and started as a UC Berkeley capstone, to generate a synthetic training data that will mitigate racial, gender or other forms of bias in datasets. We attempted to solve the problem by Generative Advesarial Network(GAN) transfomer architecture with three discirminator, one classifier and one generator.

### Installing

Pull down model from GitHub and follow virtual environment set up instructions

[Visit Project GitHub Repo](https://github.com/tflint-ucb/fair_transformer_GAN){: .btn .btn-purple }

### Getting started

Import necessary dependencies

```
from dataset import Dataset
from model import Medgan
import pandas as pd
```

Read your data into pandas
```
df = pd.read_csv('data/raw/adult.csv')
```

Create a Dataset object, which will return a numpy array that's ready for the model. This will do a couple basic [pre-processing steps] and checks to get the data ready for the model
```
my_dataset = Dataset(df)
np_input = my_dataset.pre_process(protected_col_name='gender', target_col_name='income', output_file_name='adult')
```

Initialize the model and call train for the training to begin
```
mg = Medgan()

mg.train(data=np_input,
         nEpochs=10)
```

Generate data in the following way

```
inputNum = np_input.shape[0]
mg.generateData(nSamples=inputNum)
```

[pre-processing steps]: #
