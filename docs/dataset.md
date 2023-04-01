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
> `dataset.Dataset.pre_process`(protected_col_name, y_col_name, output_file_name, multiclass=False, MinMaxScale=True)

Basic pre-processing stepd on a Pandas DataFrame.

{: .important }
> Parameters

- protected_col_name
- y_col_name
- output_file_name
- multiclass
- MinMaxScale
