# *Venture Funding with Deep Learning*
---
Alphabet Soup’s business team receives many funding applications from startups every day.  As a risk management associate at Alphabet Soup, a venture
capital firm, we will create some models that predicts whether applicants will be successful if funded by Alphabet Soup.  The business team has given
us a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety
of information about each business, including whether or not it ultimately became successful. With machine learning and neural networks, we use the
features in the provided dataset to create some binary classifier models that will predict whether an applicant will become a successful business.

We will do the following three technical deliverables:

Preprocess data for a neural network model.

Use the model-fit-predict pattern to compile and evaluate a binary classification model.

Optimize the model.

---
## Technologies:

The Jupyter file utilizes python 3.7 along with the following libraries:

pandas

pathlib

tensorflow

sklearn

math

```python
    import pandas as pd
    from pathlib import Path
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    import math
```

## Usage:
In the jupyter notebook, we begin by loading in the applicants_data.csv and review the DataFrame.
```python
    # Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
    applicant_data_df = pd.read_csv(Path("./Resources/applicants_data.csv"))

    # Review the DataFrame
    applicant_data_df
```

We will use a dataset of start-up applications to build these models.  The goal of building these models is so that we can identify which applicants will
become a successful business.  In order to build the model, all the data must be changed to integers, therefore we take the categorical variables and use
be signified by a value of `0` in the 
`loan_status` column, meaning that the loan is healthy.  A value of `1` means that the loan has a high risk of defaulting.

To build the model, first we take the CSV dataset and put it into a dataframe.  We will split the dataframe with `y` being the `loan_status` column, 
and `X` dataframe as the remaining columns.  Here we use the `value_counts` function to show us the amount of healthy loans in the dataset versus
the amount of risky loans in both the original model.  Finally, we split the data into training and testing datasets by using the function
`train_test_split`.  With the training datasets we can fit our logistic regression model.  We can predict and evaluate the model’s performance by
calculating the accuracy score of the model, generating a confusion matrix, and printing the classification report.

From the `value_counts` function we did earlier, we see that the 75036 number of good loans greatly outweight the 2500 risky loans, therefore we 
predict a new logistic regression model with resampling the training data by oversampling the high-risk loans.  We use the `RandomOverSampler`
module from the imbalanced-learn library to resample the data and with `value_counts` function we confirm that the labels have an equal number of data 
points.  Once again, we use the `LogisticRegression` classifier, but this time on the oversampled data and print and generate the same reports that we 
did from the first test in order to directly compare how our oversampled test model did.  In the next section, we will examine these results.

## Results

* Original Model:
![orig_model_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/orig_model_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.5555901527404785, Accuracy: 0.7290962338447571

* Alternative Model 1:
![alt_model1_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/alt_model1_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.5808674693107605, Accuracy: 0.7286297082901001

* Alternative Model 2:
![alt_model2_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/alt_model2_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.6244015693664551, Accuracy: 0.7287463545799255

* Alternative Model 3:
![alt_model3_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/alt_model3_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.7869622707366943, Accuracy: 0.7209329605102539

---

## Data:

The "applicants_data.csv" file is a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.
This CSV file contains a variety of information about each business, including whether or not it ultimately became successful. With machine learning
and neural networks, we use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will
become a successful business.

---

## Contributors

kevin-mau

---

## License

MIT
