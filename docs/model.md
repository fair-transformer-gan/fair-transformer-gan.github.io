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
| [`loadData`](#loadData)| Load processed data created from Dataset class |
| [`buildAutoencoder`](#buildAutoencoder) | Build autoencoder that encodes the input data |
| [`buildGenerator`](#buildGenerator) | Build the generator for training |
| [`buildGeneratorTest`](#buildGeneratorTest) | Build the generator when generating new data |
| [`getDiscriminatorResults`](#getDiscriminatorResults) | Calculate the discriminator predictions |
| [`buildDiscriminator`](#buildDiscriminator) | Build the discriminator |
| [`print2file`](#print2file) | Print the training metrics to the log file |
| [`generateData`](#generateData) | Generate new data using the trained model |
| [`calculateDiscAuc`](#calculateDiscAuc) | Calculate discriminator AUC |
| [`calculateDiscAccuracy`](#calculateDiscAccuracy) | Calculate discriminator accuracy |
| [`calculateGenAccuracy`](#calculateGenAccuracy) | Calculate generator accuracy |
| [`pair_rd`](#pair_rd) | Calculate total pairwise risk difference |
| [`calculateRD`](#calculateRD) | Calculate risk difference score across all protected attribute classes |
| [`calculateClassifierAccuracy`](#calculateClassifierAccuracy) | Calculate classifier accuracy |
| [`calculateClassifierRD`](#calculateClassifierRD) | Calculate classifier risk difference score across all protected attribute classes |
| [`create_z_masks`](#create_z_masks) | Calculate mask for each protected attribute class|
| [`train`](#train) | Train model |



### FairTransformerGAN
{: .note-title }
> CLASS
>
> `FairTransformerGAN`()


### loadData
{: .note-title }
> Method
>
> `loadData`(dataPath='')

Loads data from given path and splits it into train and validation sets.

{: .important-title }
> Parameters

- **dataPath** [str]: absolute path to processed numpy data file


{: .important-title }
> Return type

- **trainX, validX, trainz, validz, trainy, validy** [np.ndarray]: arrays of split train and validation data arrays of split train and validation data



### buildAutoencoder
{: .note-title }
> Method
>
> `buildAutoencoder`(x_input)

Builds the autoencoder that encodes, compresses and then decompresses the input. Calculates the loss between the decompressed input and the original input.

{: .important-title }
> Parameters

- **x_input** [tf.Tensor]: tensor of x input data


{: .important-title }
> Return type

- **loss** [tf.Tensor]: float tensor loss between the decompressed input and the original x input
- **decodeVariables** [dict]: variable that stores weights and biases of decompressed x input



### buildGenerator
{: .note-title }
> Method
>
> `buildGenerator`(x_input, y_input, z_input, bn_train)

Builds the generator. Generates the x data given the y outcome and z protected attribute during training. Applies multi-head self-attention to x data using MultiHeadSelfAttention class.

{: .important-title }
> Parameters

- **x_input** [tf.Tensor]: tensor of x input data
- **y_input** [tf.Tensor]: tensor of y outcome data
- **z_input** [tf.Tensor]: tensor of z protected attribute
- **bn_train** [tf.Tensor]: boolean tensor specifying whether we are in the training phase of generator


{: .important-title }
> Return type

- **output** [tf.Tensor]: generated x data




### buildGeneratorTest
{: .note-title }
> Method
>
> `buildGenerator`(x_input, y_input, z_input, bn_train)

Builds the generator for post model training. Generates the x data given the y outcome and z protected attribute post model training. Applies multi-head self-attention to x data using MultiHeadSelfAttention class.

{: .important-title }
> Parameters

- **x_input** [tf.Tensor]: tensor of x input data
- **y_input** [tf.Tensor]: tensor of y outcome data
- **z_input** [tf.Tensor]: tensor of z protected attribute
- **bn_train** [tf.Tensor]: boolean tensor specifying whether we are in the training phase of generator


{: .important-title }
> Return type

- **output** [tf.Tensor]: generated x data


### getDiscriminatorResults
{: .note-title }
> Method
>
> `getDiscriminatorResults`(x_input, y_bool, keepRate, z_mask0, z_mask1, z_mask2, z_mask3, reuse=False)

Calculates the discriminator predictions.

{: .important-title }
> Parameters

- **x_input** [tf.Tensor]: tensor of x input data
- **y_bool** [tf.Tensor]: boolean tensor representing the boolean value of outcome y 
- **keepRate** [float]: dropout rate of discriminator
- **z_mask0** [tf.Tensor]: boolean tensor which is True where the z protected attribute is 0
- **z_mask1** [tf.Tensor]: boolean tensor which is True where the z protected attribute is 1
- **z_mask2** [tf.Tensor]: boolean tensor which is True where the z protected attribute is 2
- **z_mask3** [tf.Tensor]: boolean tensor which is True where the z protected attribute is 3
- **reuse** [bool]: whether or not to reuse TensorFlow variables

{: .important-title }
> Return type

- **f_hat** [tf.Tensor]: probabilities of generated x data being real/fake based the protected attribute z
- **y_hat** [tf.Tensor]: probabilities of generated y classified by classifier based on the generated x data 
- **z_hat** [tf.Tensor]: probabilities of protected attribute z based based on generated y
- **g_hat** [tf.Tensor]: probabilities of generated x input 



### buildDiscriminator
{: .note-title }
> Method
>
> `buildDiscriminator`(x_real, y_real, x_fake, y_fake, yb_real, yb_fake, keepRate, decodeVariables, z_r_mask0, z_r_mask1, z_r_mask2, z_r_mask3, z_r_mask4, z_f_mask0, z_f_mask1, z_f_mask2, z_f_mask3, z_f_mask4)

Builds the discriminator.

{: .important-title }
> Parameters

- **x_real** [tf.Tensor]: tensor of x real input 
- **y_real** [tf.Tensor]: tensor of y real outcome 
- **x_fake** [tf.Tensor]: tensor of x fake input 
- **y_fake** [tf.Tensor]: tensor of y fake outcome 
- **yb_real** [tf.Tensor]: boolean tensor which is True where the real y real outcome is 0
- **yb_fake** [tf.Tensor]: boolean tensor which is True where the real y fake outcome is 0
- **keepRate** [float]: dropout rate of discriminator
- **decodeVariables** [dict]: variable that stores weights and biases of decompressed x input
- **z_r_mask0** [tf.Tensor]: boolean tensor which is True where the real z protected attribute is 0
- **z_r_mask1** [tf.Tensor]: boolean tensor which is True where the real z protected attribute is 1
- **z_r_mask2** [tf.Tensor]: boolean tensor which is True where the real z protected attribute is 2
- **z_r_mask3** [tf.Tensor]: boolean tensor which is True where the real z protected attribute is 3
- **z_r_mask4** [tf.Tensor]: boolean tensor which is True where the real z protected attribute is 4
- **z_f_mask0** [tf.Tensor]: boolean tensor which is True where the fake z protected attribute is 0
- **z_f_mask1** [tf.Tensor]: boolean tensor which is True where the fake z protected attribute is 1
- **z_f_mask2** [tf.Tensor]: boolean tensor which is True where the fake z protected attribute is 2
- **z_f_mask3** [tf.Tensor]: boolean tensor which is True where the fake z protected attribute is 3
- **z_f_mask4** [tf.Tensor]: boolean tensor which is True where the fake z protected attribute is 4


{: .important-title }
> Return type

-  **tensors** [tf.Tensors]: decoded/predicted x, losses and probabilities of real and fake variables f, y, z, and g


### print2file
{: .note-title }
> Method
>
> `print2file`(buf, outFile)

Writes training metrics to log file.

{: .important-title }
> Parameters

- **buf** [str]: data to write to file
- **outFile** [str]: file path to model output


{: .important-title }
> Return type

[None]


### generateData
{: .note-title }
> Method
>
> `generateData`(nSamples=100,modelFile='model',batchSize=100, outFile='out', p_z=[], p_y=[])

Generates less-biased data using trained model and save to file.

{: .important-title }
> Parameters

- **nSamples** [int]: size of entire original dataset
- **modelFile** [str]: path to trained Fair Transformer GAN model
- **batchSize** [int]: size each batch
- **outFile** [str]: path to generated data files in numpy format
- **p_z** [list]: probability distribution of protected attribute
- **p_y** [list]: probability distribution of outcome

{: .important-title }
> Return type

[None]


### calculateDiscAuc
{: .note-title }
> Method
>
> `calculateDiscAuc`(preds_real, preds_fake)

Calculates discriminator AUC from real and fake predictions

{: .important-title }
> Parameters

- **preds_real** [numpy.ndarray]: array of real predictions
- **preds_fake** [numpy.ndarray]: array of fake predictions


{: .important-title }
> Return type

- **auc** [float]: discriminator AUC



### calculateDiscAccuracy
{: .note-title }
> Method
>
> `calculateDiscAccuracy`(preds_real, preds_fake)

Calculates discriminator accuracy from real and fake predictions

{: .important-title }
> Parameters

- **preds_real** [numpy.ndarray]: array of real predictions
- **preds_fake** [numpy.ndarray]: array of fake predictions


{: .important-title }
> Return type

- **acc** [float]: discriminator accuracy

### calculateGenAccuracy
{: .note-title }
> Method
>
> `calculateGenAccuracy`(preds_real, preds_fake)

Calculates generator accuracy from real and fake predictions

{: .important-title }
> Parameters

- **preds_real** [numpy.ndarray]: array of real predictions
- **preds_fake** [numpy.ndarray]: array of fake predictions


{: .important-title }
> Return type

- **acc** [float]: generator accuracy

### pair_rd
{: .note-title }
> Method
>
> `pair_rd`(y_real, z_real)

Helper function to calculate total pairwise risk difference across all z protected attribute classes

{: .important-title }
> Parameters

- **y_real** [numpy.ndarray]: array of y outcome values
- **z_real** [numpy.ndarray]: array of z protected attribute values 


{: .important-title }
> Return type

- **risk_diff** [float]: total risk difference score across all z protected attribute classes


### calculateRD
{: .note-title }
> Method
>
> `calculateRD`(y_real, z_real)

Calculates risk difference score across all z protected attribute classes during training. Calls pair_rd() function. 


{: .important-title }
> Parameters

- **y_real** [numpy.ndarray]: array of original y outcome values
- **z_real** [numpy.ndarray]: array of original z protected attribute values 


{: .important-title }
> Return type

- **risk_diff** [float]: total risk difference score across all z protected attribute classes


### calculateClassifierAccuracy
{: .note-title }
> Method
>
> `calculateClassifierAccuracy`(preds_real, y_real)

Calculates classifier accuracy between real y outcome and predicted y.


{: .important-title }
> Parameters

- **preds_real** [numpy.ndarray]: array of predicted y based on x data generated from real x data
- **y_real** [numpy.ndarray]: array of original y outcome values


{: .important-title }
> Return type

- **acc** [float]: classider accuracy



### calculateClassifierRD
{: .note-title }
> Method
>
> `calculateClassifierRD`(preds_real, z_real, yreal)

Calculate classifier risk difference score across all z protected attribute classes during training.


{: .important-title }
> Parameters

- **preds_real** [numpy.ndarray]: array of predicted y based on x data generated from real x data
- **z_real** [numpy.ndarray]: array of original z protected attribute values 
- **y_real** [numpy.ndarray]: array of original y outcome values



{: .important-title }
> Return type

- **rd** [float]: total risk difference score across all z protected attribute classes
- **rd1** [float]: risk difference score across all z protected attribute classes when y outcome = 1
- **rd0** [float]: risk difference score across all z protected attribute classes when y outcome = 0




### create_z_masks
{: .note-title }
> Method
>
> `create_z_masks`(z_arr)

Create a z_mask for each class (max 5) of protected attribute in z array. Each boolean mask is True at each index it exists in the z array.


{: .important-title }
> Parameters

- **z_arr** [numpy.ndarray]: array of z protected attribute values 



{: .important-title }
> Return type

- **z_mask0** [numpy.ndarray]: array of z_mask for protected attribute class 0
- **z_mask1** [numpy.ndarray]: array of z_mask for protected attribute class 1
- **z_mask2** [numpy.ndarray]: array of z_mask for protected attribute class 2
- **z_mask3** [numpy.ndarray]: array of z_mask for protected attribute class 3
- **z_mask4** [numpy.ndarray]: array of z_mask for protected attribute class 4



### train
{: .note-title }
> Method
>
> `create_z_masks`(dataPath='data',
              modelPath='',
              outPath='out',
              pretrainEpochs=100,
              nEpochs=300,
              generatorTrainPeriod=1,
              discriminatorTrainPeriod=2,
              pretrainBatchSize=100,
              batchSize=1000,
              saveMaxKeep=0, p_z=[], p_y=[])

Train the Fair Transformer GAN model and save output.

{: .important-title }
> Parameters

- **dataPath** [str]: path to input dataset 
- **modelPath** [str]: string to existing model, if it exists
- **outPath** [str: path to store model output and logs
- **nEpochs** [int]: number of epochs to train the model
- **discriminatorTrainPeriod** [int]: number of periods to train discriminator 
- **generatorTrainPeriod** [int]: number of periods to train generator 
- **pretrainBatchSize** [int]: size of pretraining batch
- **batchSize** [int]: size of training batch
- **pretrainEpochs** [int: number of epochs to pretrain model
- **saveMaxKeep** [int]: number of checkpoint files to save
- **p_z** [list]: probability distribution of protected attribute
- **p_y** [list]: probability distribution of outcome

{: .important-title }
> Return type

[None]