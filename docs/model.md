---
layout: default
title: Model
parent: API
nav_order: 2
---

### Fair Transformer GAN class

A class that defines the Fair Transformer GAN model. Processed data from the Dataset object is passed to the Model object, which trains the model and generates less-biased data. The Metrics object then uses this generated data to calculate model performance. 

The train folder contains training script that can be run via command line.

| Function     | Description      |
|:-------------|:------------------|
| [`loadData`](#datasetdatasetpre_process)| Load processed data created from Dataset class |
| [`buildAutoencoder`](#datasetdatasetpost_process) | Build autoencoder that encodes the input data |
| [`buildGenerator`](#datasetdatasetget_protected_distribution) | Build the generator for training |
| [`buildGeneratorTest`](#datasetdatasetget_target_distribution) | Build the generator when generating new data |
| [`getDiscriminatorResults`](#datasetdatasetget_target_distribution) | Calculate the discriminator predictions |
| [`buildDiscriminator`](#datasetdatasetget_protected_distribution) | Build the discriminator |
| [`print2file`](#datasetdatasetget_target_distribution) | Print the training metrics to the log file |
| [`generateData`](#datasetdatasetget_target_distribution) | Generate new data using the trained model |
| [`calculateDiscAuc`](#datasetdatasetget_target_distribution) | Calculate discriminator AUC |
| [`calculateDiscAccuracy`](#datasetdatasetget_target_distribution) | Calculate discriminator accuracy |
| [`calculateGenAccuracy`](#datasetdatasetget_target_distribution) | Calculate generator accuracy |
| [`pair_rd`](#datasetdatasetget_target_distribution) | Calculate total pairwise risk difference |
| [`calculateRD`](#datasetdatasetget_target_distribution) | Calculate risk difference score across all protected attribute classes |
| [`calculateClassifierAccuracy`](#datasetdatasetget_target_distribution) | Calculate classifier accuracy |
| [`calculateClassifierRD`](#datasetdatasetget_target_distribution) | Calculate classifier risk difference score across all protected attribute classes |
| [`create_z_masks`](#datasetdatasetget_target_distribution) | Calculate mask for each protected attribute class|
| [`train`](#datasetdatasetget_target_distribution) | Train models |



