---
layout: default
title: Metrics
parent: API
nav_order: 3
---

### Metrics class

| Function     | Description      |
|:-------------|:------------------|
| [`binary_fair_data_generation_metrics`](#Metricsbinary_fair_data_generation_metrics)| This method is used to evaluate the fairness of the generated/synthetic data. This method calculates the following fairness metrics: Risk Difference (RD), Balanced Error Rate (BER), and Euclidean Distance (with and without the protected attribute S). RD is the difference in the probability of the positive outcomes between two groups defined by S and is meant to capture disparate treatment. BER is the average error rate of predicting the protected attribute S using X features and is meant to capture disparate impact. The Euclidean Distance metrics measure the distance between the joint probability distributions of the synthetic and real datasets, with and without the protected attribute S, respectively. The distances represent the closeness and faithfulness to the original dataset. |
| [`multi_fair_data_generation_metrics`](#Metricsmulti_fair_data_generation_metrics) | This method is similar to the binary version, but it is designed to handle multiple protected attributes. The main difference is that it calculates a relaxed version of the Risk Difference metric. It first calculates the probability of the positive outcome for each group defined by the protected attribute S. Then, it checks whether these probabilities are within one or two standard deviations of the mean probability of the positive outcome across all groups. If they are, then the metric is deemed to be fair, and its value is set to TRUE. If not, then the metric is set to FALSE. This metric is more lenient than the original Risk Difference metric since it allows for some variation in the probability of the positive outcome across different groups. |
| [`binary_fair_data_classification_metrics`](#Metricsbinary_fair_data_classification_metrics) | This method is used to evaluate the fairness of binary classification models on given data. This method calculates the following fairness metrics: Demographic Parity (DP), Difference in True Positive Rates (DTPR), Difference in False Positive Rates (DFPR), Accuracy, and F1 Score. DP is meant to reflect whether a model trained on the generated data learns to predict fairer outcomes. DTPR and DFPR and meant to capture equality of odds so that the model is not systemically advantaging or disadvantaging any particular group. Finally, accuracy and F1 score are measured to see if there is a tradeoff in utility compared to a model trained on the original dataset (we expect a decline in utility in exchange for fairness). |
| [`multi_fair_data_classification_metrics`](#Metricsmulti_fair_data_classification_metrics) | This method is similar to the binary version, but it is designed to handle multiple protected attributes. Similar to the multi-fair data generation method, it calculates a relaxed version of Demographic Parity, Difference in True Positive Rates, and Difference in False Positive Rates using the same logic. This metric is more lenient than the original metrics since it allows for some variation in the metrics across different groups. |


### Metrics
{: .note-title }
> CLASS
>
> `Metrics`()


### Metrics.binary_fair_data_generation_metrics
{: .note-title }
> Method
>
> `Metrics.binary_fair_data_generation_metrics`(df, df_real)

Calculate fair data generation metrics for binary protected attributes.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN
- **df_real** [DataFrame] - The pre-processed dataset containing the original/real data

{: .important-title }
> Return type

Dictionary with the fair data generation metrics.



### Metrics.multi_fair_data_generation_metrics
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

### Metrics.binary_fair_data_classification_metrics
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


### Metrics.multi_fair_data_classification_metrics
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
