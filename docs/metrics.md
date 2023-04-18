---
layout: default
title: Metrics
parent: API
nav_order: 3
---

### Metrics class

| Function     | Description       |
|:-------------|:------------------|
| [`binary_fair_data_generation_metrics`](#binary_fair_data_generation_metrics)| This method is used to evaluate the fairness of the generated/synthetic data. This method calculates the following fairness metrics: Risk Difference (RD) and Balanced Error Rate (BER). RD is the difference in the probability of the positive outcomes between two protected groups and is meant to capture disparate treatment. BER is the average error rate of predicting the protected attribute using X features and is meant to capture disparate impact. |
| [`multi_fair_data_generation_metrics`](#multi_fair_data_generation_metrics) | This method is similar to the binary version, but it is designed to handle multiple protected attributes. The main difference is that it calculates a relaxed version of the Risk Difference metric. It first calculates the probability of the positive outcome for each group defined by the protected attribute. Then, it checks whether these probabilities are within a certain number of standard deviations of the mean probability of the positive outcome across all groups. If they are, then the metric is deemed to be fair, and its value is set to TRUE. If not, then the metric is set to FALSE. This metric is more lenient than the original Risk Difference metric since it allows for some variation in the probability of the positive outcome across different groups. |
| [`euclidean_distance`](#metricseuclidean_distance) | The Euclidean Distance metrics measure the distance between the joint probability distributions of the synthetic and real datasets, with and without the protected attribute, respectively. The distances represent the closeness and faithfulness to the original dataset. |
| [`binary_fair_classification_metrics`](#binary_fair_classification_metrics) | This method is used to evaluate the fairness of binary classification models on given data. This method calculates the following fairness metrics: Demographic Parity (DP), Difference in True Positive Rates (DTPR), Difference in False Positive Rates (DFPR), Accuracy, and F1 Score. DP is meant to reflect whether a model trained on the generated data learns to predict fairer outcomes. DTPR and DFPR and meant to capture equality of odds so that the model is not systemically advantaging or disadvantaging any particular group. Finally, accuracy and F1 score are measured to see if there is a tradeoff in utility compared to a model trained on the original dataset (we expect a decline in utility in exchange for fairness). |
| [`multi_fair_classification_metrics`](#multi_fair_classification_metrics) | This method is similar to the binary version, but it is designed to handle multiple protected attributes. Similar to the multi-fair data generation method, it calculates a relaxed version of Demographic Parity, Difference in True Positive Rates, and Difference in False Positive Rates using the same logic. This metric is more lenient than the original metrics since it allows for some variation in the metrics across different groups. |


### Metrics
{: .note-title }
> CLASS
>
> `Metrics`()


### binary_fair_data_generation_metrics
{: .note-title }
> Method
>
> `binary_fair_data_generation_metrics(df)`

Calculate fair data generation metrics for binary protected attributes.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN

{: .important-title }
> Return type

Dictionary with the fair data generation metrics.

### multi_fair_data_generation_metrics
{: .note-title }
> Method
>
> `multi_fair_data_generation_metrics(df)`

Calculate fair data generation metrics for multiple protected attributes.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN

{: .important-title }
> Return type

Dictionary with the fair data generation metrics.

### euclidean_distance
{: .note-title }
> Method
>
> `euclidean_distance(df, df_real)`

Calculate euclidean distance of joint probability distributions between two datasets.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN
- **df_real** [DataFrame] - The pre-processed dataset containing the original/real data

{: .important-title }
> Return type

Dictionary with the distance metrics.

### binary_fair_classification_metrics
{: .note-title }
> Method
>
> `binary_fair_classification_metrics(X_real, y_real, y_pred)`

Calculate fair data classification metrics for binary protected attributes.

{: .important-title }
> Parameters

- **X_real** [DataFrame] - The original/real X features
- **y_real** [DataFrame] - The original/real y outcomes
- **y_pred** [DataFrame] - The predictions from a classification model trained on the generated/synthetic dataset and predicted on X_real

{: .important-title }
> Return type

Dictionary with the fair data classification metrics.

### multi_fair_classification_metrics
{: .note-title }
> Method
>
> `binary_fair_classification_metrics(X_real, y_real, y_pred)`

Calculate fair data classification metrics for multiple protected attributes.

{: .important-title }
> Parameters

- **X_real** [DataFrame] - The original/real X features
- **y_real** [DataFrame] - The original/real y outcomes
- **y_pred** [DataFrame] - The predictions from a classification model trained on the generated/synthetic dataset and predicted on X_real

{: .important-title }
> Return type

Dictionary with the fair data classification metrics.

### Fair Data Generation Metrics

| Metric       | Formula    | 
|:-------------|:------------------|
| Risk Difference | [`RD = P(Y = 1|Z = 1) − P(Y = 1|Z = 0)`] |
| Balanced Error Rate | [`BER(f(X), Z) = P(f(X) = 0 | Z = 1) + P(f(X) = 1 | Z = 0) / 2`] |

### Data Faithfulness (Euclidean Distance) Metrics

| Metric       | Formula    | 
| Euclidean Distance (with the protected attribute) | [`dist(X, Y, Z) = ||Pdata(X, Y, Z)−PG(X, Y, Z)||`] |
| Euclidean Distance (without the protected attribute) | [`dist(X, Y) = ||Pdata(X, Y ) − PG(X, Y )||`] |

### Fair Data Classification Metrics

| Metric       | Formula    | 
| Demographic Parity | [`RD(η) = P(η(X) = 1|Z = 1) − P(η(X) = 1|Z = 0)`] |
| Difference in True Positive Rates | [`P(η(X) = 1|Y =1, Z = 1) − P(η(X) = 1|Y = 1, Z = 0)`] |
| Difference in False Positive Rates | [`P(η(X) = 1|Y = 0, Z = 1)−P(η(X) = 1|Y = 0, Z = 0)`] |

### Data Utility Metrics

| Metric       | Formula    | 
| Accuracy | [`(True Positives + True Negatives) / Total Predictions`] |
| F1 score | [`(2 * Precision * Recall) / (Precision + Recall)`] |
