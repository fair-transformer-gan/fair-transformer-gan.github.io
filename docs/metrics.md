---
layout: default
title: Metrics
parent: API
nav_order: 3
---

### Metrics class

| Function     | Description      |
|:-------------|:------------------|
| [`binary_fair_data_generation_metrics`](#datasetdatasetpre_process)| This method calculates four fairness metrics: Risk Difference, Balanced Error Rate, Distance w/o S, and Distance with S. The first metric is the difference in the probability of the positive outcome (i.e., 1) between two groups defined by a protected attribute (S). The second metric is the average error rate of predicting the positive outcome for each group, weighted by the size of each group. The third and fourth metrics measure the distance between the joint probability distributions of the synthetic and real datasets, with and without the protected attribute S, respectively. |
| [`multi_fair_data_generation_metrics`](#datasetdatasetpost_process) | This method calculates a relaxed version of the Risk Difference metric. It first calculates the probability of the positive outcome for each group defined by the protected attribute S. Then, it checks whether these probabilities are within one or two standard deviations of the mean probability of the positive outcome across all groups. If they are, then the metric is deemed to be fair, and its value is set to 0. If not, then the metric is set to the average difference between the probability of the positive outcome in each group and the mean probability of the positive outcome across all groups. This metric is more lenient than the original Risk Difference metric since it allows for some variation in the probability of the positive outcome across different groups. |
| [`binary_fair_data_classification_metrics`](#datasetdatasetget_protected_distribution) | This method is used to evaluate the fairness of binary classification models on given data. The input to this method is the predicted labels and true labels for the binary classification task, along with the sensitive attribute values of the individuals in the dataset. The method returns various fairness metrics such as disparate impact, equal opportunity difference, and demographic parity difference. Disparate impact measures the ratio of favorable outcomes for the unprivileged group to that of the privileged group. Equal opportunity difference measures the difference in true positive rate between the unprivileged and privileged groups, while demographic parity difference measures the difference in positive rate between the unprivileged and privileged groups. |
| [`multi_fair_data_classification_metrics`](#datasetdatasetget_target_distribution) | This method is similar to the binary_fair_data_classification_metrics method, but is used for evaluating the fairness of multi-class classification models. The input to this method is again the predicted labels and true labels, along with the sensitive attribute values. The method returns various fairness metrics such as mean demographic disparity, mean equal opportunity difference, and mean conditional demographic disparity. Mean demographic disparity measures the difference in positive rate across all classes between the unprivileged and privileged groups. Mean equal opportunity difference measures the difference in true positive rate across all classes between the unprivileged and privileged groups. Mean conditional demographic disparity measures the difference in positive rate for each class between the unprivileged and privileged groups. |


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
