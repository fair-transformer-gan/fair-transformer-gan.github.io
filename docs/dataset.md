---
layout: default
title: Dataset
parent: API
nav_order: 1
---

### Dataset class

A helper class for data pre-processing. Automates steps such as binarizing and scaling data, placing target and protected class in the right position, checks for nulls and allowed column types.

{: .important }
> This is an optional step, if passing pre-processed data into the [Model](/docs/model.html) class make sure to place the protected variable column first and target variable column last.

| Function     | Description      |
|:-------------|:------------------|
| [`init`](#init)| Intializes instance of Dataset class |
| [`pre_process`](#pre_process)| This step is optional and does a couple simple pre-processing check and adjustments, such as scaling, checking for nulls, setting protected and target columns |
| [`post_process`](#post_process) | Run on generated dataset to inverse scaling |
| [`get_protected_distribution`](#get_protected_distribution)           | Gets protected variable distribution from the dataset |
| [`get_target_distribution`](#get_target_distribution)           | Gets target variable distribution from the dataset |



### Dataset
{: .note-title }
> CLASS
>
> `Dataset`()

### init
{: .note-title }
> Method
>

`__init__(dataframe)`
Initializes dataset class given a Pandas Dataframe.

{: .important-title }
> Parameters

- **dataframe** [Pandas DataFrame]: Training data before pre-processing
- **scaler** [sklearn.preprocessing.MinMaxScaler]: initialize the MinMaxScaler object that will be used to scale data in pre-processing and post-processing
- **np_data** [numpy.ndarray]: Processed data in a Numpy array
- **target_names** List[str]: Target variable column names
- **protected_names** List[str]: Protected variable column names
- **processed_col_types** List[str]: store list for the original dataset column types, used in post-processing


{: .important-title }
> Return type
>
>`None`


### pre_process
{: .note-title }
> Method
>
> `pre_process(protected_var, outcome_var, output_file_name_path, multiclass=False, min_max_scale=True)`

Basic pre-processing on a Pandas DataFrame including one-hot encoding, scaling, checking for nulls, etc. Saves a pickle file with a numpy array and a csv file with a data dictionary in the specified path.


{: .important-title }
> Parameters

- **protected_var** [str] - Name of the protected column in the Pandas DataFrame
- **outcome_var** [str] - Name of the outcome column in the Pandas DataFrame
- **output_file_name_path** [str] - Name the Pickle file that will be saved in the data/interim folder
- **multiclass** (Optional [bool]) - Set to True if your protected variable is categorical has more than two states.
- **min_max_scale** (Optional [bool]) - Set to False if using scaled data

{: .important-title }
> Raises

**Exception** - if dataset has nulls

{: .important-title }
> Return type

- **np_data** [numpy.ndarray]: numpy array with pre-processed data


### post_process
{: .note-title }
> Method
>
> `post_process(gen_data_np)`

Inverse scaling on the generated data from the trained model.

{: .important-title }
> Parameters

- **gen_data_np** [np.ndarray] - numpy array with generated data

{: .important-title }
> Return type

- **gen_data_np** [numpy.ndarray]: numpy array with post-processed data


### get_protected_distribution
{: .note-title }
> Method
>
> `get_protected_distribution(np_data)`

Calculates the protected variable distribution after pre-processing. 

{: .important-title }
> Parameters

- **np_data** [np.ndarray] - data in a numpy array

{: .important-title }
> Return type

- **protected_distribution** [List[float]]: distrbution of each protected class

To get the dataframe columns corresponding to class names, run `protected_names` on the Dataset object. For example

```
dataset = Dataset()
np_data = dataset.pre_process(df, 'gender', 'income', 'out_file')

dataset.get_protected_distribution(np_data)

>>> [85.6, 14.4]

dataset.protected_names

>>> ['Male', 'Female']

```


### get_target_distribution
{: .note-title }
> Method
>
> `get_target_distribution(np_data)`

Returns the target variable distribution after pre-processing. 

{: .important-title }
> Parameters

- **np_data** [np.ndarray] - data in a numpy array

{: .important-title }
> Return type

- **target_distribution** [List[float]]: distrbution of each target class


{: .note }
> To get the dataframe columns corresponding to class names, return `target_names` on the Dataset object.



