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
`OneHotEncoder` to convert the DataFrame.  Then, we use the encoded DataFrame and concatenate it with the numerical variables from the original DataFrame.  
```python
    # Create a list of categorical variables 
    categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes == "object"].index)
    
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)
    
    # Encode the categorcal variables using OneHotEncoder
    encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])
    
    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(encoded_data, columns = enc.get_feature_names(categorical_variables))
    
    # Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
    numerical_variables_df = applicant_data_df.drop(columns = categorical_variables)
    encoded_df = pd.concat(
        [
            numerical_variables_df,
            encoded_df
        ],
        axis=1
    )

```

With this encoded DataFrame, we will create the features (X) and target (y) datasets for use in our model to use as our training and testing datasets.  The
target dataset is defined by the preprocessed DataFrame column “IS_SUCCESSFUL”.  The remaining columns define the features dataset.
```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

Use scikit-learn's `StandardScaler` to scale the features data
```python
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler to the features training dataset
    X_scaler = scaler.fit(X_train)

    # Fit the scaler to the features training dataset
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
```

We will now use the model-fit-predict pattern to compile and evaluate a binary classification model.

Model 1: 116 input features & 2 hidden layers.  1st hidden layer has 58 nodes, 2nd hidden layer has 29 nodes.  Using the Sequential model and relu activation.
We compile the model and fit the model using 50 epochs.
```python
    # Compile the Sequential model
    nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model using 50 epochs and the training data
    fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
```

Below we will review the results of this model, but first let's try to optimize the model.

Alternate Model 1:  This time we will try it with more hidden layers.  We will try 4 hidden layers, starting at 58 nodes, down to 29, then 15, and finally 8 as
we go each layer.  Again we will use relu but this time we will try linear activation for the outer layer.  Also, instead of 50 epochs, we'll try 800.

Alternate Model 2:  This time we will try with a different number of nodes in the hidden layer.  The number of nodes will be the square root of sum of the number
of input features and number of output neurons.

Alternate Model 3:  This time we will try the original model again, except we will exclude two DataFrame columns: “STATUS” and “SPECIAL_CONSIDERATIONS” as these 
two features columns may have variables or outliers that might be confusing the model.

## Results

Original Model:


![orig_model_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/orig_model_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.5555901527404785, Accuracy: 0.7290962338447571

Alternative Model 1:


![alt_model1_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/alt_model1_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.5808674693107605, Accuracy: 0.7286297082901001

Alternative Model 2:


![alt_model2_summary](https://github.com/kevin-mau/venture_funding_with_deep_learning/blob/main/Resources/alt_model2_summary.PNG?raw=true)
  * The model loss and accuracy results: Loss: 0.6244015693664551, Accuracy: 0.7287463545799255

Alternative Model 3:


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
