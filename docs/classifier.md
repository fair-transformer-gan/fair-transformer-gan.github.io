---
layout: default
title: Classifier
parent: API
nav_order: 4
---

### Classifier class

A class that trains a basic logistic regression or random forest model (as used in our testing) and returns the inputs required for the fair classification metrics method.

Note: Users do not have to use this class and can train their own classifiers.

| Function     | Description       |
|:-------------|:------------------|
| [`logistic_regression`](#logistic_regression)| Trains a logistic regression model and returns real X data, real y data, and the predicted y data. |
| [`random_forest`](#random_forest) | Trains a random forest model and returns real X data, real y data, and the predicted y data. |


### Classifier
{: .note-title }
> CLASS
>
> `Classifier`()


### logistic_regression
{: .note-title }
> Method
>
> `logistic_regression(df, df_real)`

Fit a logistic regression model on df and predict on df_real. Split df_real into X data and y data.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN
- **df_real** [DataFrame] - The original dataset

{: .important-title }
> Return type

Returns real x data from df_real, real y data from df_real, and the predicted y data from the logistic regression model predictions. Can be fed into the fair classification method from the Metrics() class.

### random_forest
{: .note-title }
> Method
>
> `random_forest(df, df_real)`

Fit a random forest model on df and predict on df_real. Split df_real into X data and y data.

{: .important-title }
> Parameters

- **df** [DataFrame] - The synthetic/generated dataset using Fair Transformer GAN
- **df_real** [DataFrame] - The original dataset

{: .important-title }
> Return type

Returns real x data from df_real, real y data from df_real, and the predicted y data from the random forest model predictions. Can be fed into the fair classification method from the Metrics() class.
