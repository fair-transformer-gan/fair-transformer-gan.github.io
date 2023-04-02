---
layout: default
title: Dataset
parent: API
nav_order: 1
---

### Dataset class

| Function     | Descriptioin      |
|:-------------|:------------------|
| `pre_process`| This step is optional and does a couple simple pre-processing check and adjustments, such as scaling, checking for nulls, setting protected and target columns |
| `post_process` | Run on generated dataset to inverse scaling |
| `get_protected_distribution`           | Gets protected variable distribution from the dataset |
| `get_target_distribution`           | Gets target variable distribution from the dataset |


#### dataset.Dataset
{: .note-title }
> CLASS
>
> `dataset.Dataset.pre_process`(protected_col_name, y_col_name, output_file_name, multiclass=False, min_max_scale=True)

Basic pre-processing stepd on a Pandas DataFrame.

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