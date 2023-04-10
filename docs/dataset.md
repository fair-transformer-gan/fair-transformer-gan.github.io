---
layout: default
title: Dataset
parent: API
nav_order: 1
---

### Dataset class

A helper class for data pre-processing. Automates steps such as binarizing and scaling data, placing target and protected class in the right position, checks for nulls and allowed column types.

| Function     | Descriptioin      |
|:-------------|:------------------|
| [`pre_process`](#datasetdatasetpre_process)| This step is optional and does a couple simple pre-processing check and adjustments, such as scaling, checking for nulls, setting protected and target columns |
| [`post_process`](#datasetdatasetpost_process) | Run on generated dataset to inverse scaling |
| [`get_protected_distribution`](#datasetdatasetget_protected_distribution)           | Gets protected variable distribution from the dataset |
| [`get_target_distribution`](#datasetdatasetget_target_distribution)           | Gets target variable distribution from the dataset |


### dataset.Dataset
{: .note-title }
> CLASS
>
> `dataset.Dataset`()


### dataset.Dataset.pre_process
{: .note-title }
> Method
>
> `dataset.Dataset.pre_process`(protected_col_name, y_col_name, output_file_name, multiclass=False, min_max_scale=True)

Basic pre-processing on a Pandas DataFrame.

{: .important-title }
> Parameters

- **protected_col_name** [str] - Name of the protected column in the Pandas DataFrame
- **y_col_name** [str] - Name of the target column in the Pandas DataFrame
- **output_file_name** [str] - Name the Pickle file that will be saved in the data/interim folder
- **multiclass** (Optional [bool]) - Set to True if your protected variable is categorical has more than two states.
- **min_max_scale** (Optional [bool]) - Set to False if using scaled data

{: .important-title }
> Raises

**Exception** - if dataset has nulls

{: .important-title }
> Return type

Numpy array with pre-processed data



### dataset.Dataset.post_process
{: .note-title }
> Method
>
> `dataset.Dataset.post_process`(gen_data_np)

Iverse scaling on the generated dataset.

{: .important-title }
> Parameters

- **gen_data_np** [np.ndarray] - Numpy array with generated data.

{: .important-title }
> Return type

Numpy array with post-processed data

### dataset.Dataset.get_protected_distribution
{: .note-title }
> Method
>
> `dataset.Dataset.get_protected_distribution`

Returns the protected variable distribution after pre-processing. 

{: .important-title }
> Parameters

`None`

{: .important-title }
> Return type

`List[float]` - distrbution of each protected class

To get the dataframe columns corresponding to class names, run `protected_names` on the Dataset object. For example

```
dataset = Dataset()
dataset.pre_process(df, 'gender', 'income', 'out_file')

dataset.get_protected_distribution()

>>> [85.6, 14.4]

dataset.protected_names

>>> ['Male', 'Female']

```


### dataset.Dataset.get_target_distribution
{: .note-title }
> Method
>
> `dataset.Dataset.get_target_distribution`

Returns the target variable distribution after pre-processing. 

{: .important-title }
> Parameters

`None`

{: .important-title }
> Return type

`List[float]` - distrbution of each target class

To get the dataframe columns corresponding to class names, return `target_names` on the Dataset object.
