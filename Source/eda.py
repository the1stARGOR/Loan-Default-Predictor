import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler


# Transform the data by replacing null values in specific columns with the string "missing" using the fillna() function.
def replace_null_values_with_a_value(df, columns,value):
    # Replace null values with "missing" in specific columns
    for column in columns:
        df[column] = df[column].fillna(value)
    return df


# Print number of unique values in all columns. This is going to give us the idea of what we are dealing with.
def unique_values_each_column(df):
    # Print number of unique values in all columns
    for col in df.columns:
        print(col, ':', df[col].nunique())


#Function to drop columns
def drop_columns(df, columns_to_drop):
    for column in columns_to_drop:
      df.drop(column, axis=1, inplace=True)

    return df


# Display feature distributions histogram
def Feature_Distributions_Histogram(df):
    # Generate histograms for each column
    for c in df.columns:
        fig = px.histogram(df, x=c, hover_data=df.columns)
        fig.show()


# Display pearson correlation matrix
def correlation_heatmap(train):
    correlations = train.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();


# Let's plot all interaction scatter plots using seaborn
def plot_pairwise_scatter(df):
    # Get the list of numeric features
    features = df.select_dtypes(include='number').columns

    # Create the pairwise scatter plots using Seaborn
    sns.pairplot(df[features])


import plotly.express as px
import numpy as np



# Fix skewness in the datframe
def fix_skewness(df, features):
  features_log = features
  for f in features_log:
    sk= df[f].skew()
    print("Inital skewness in feature: ",f," is: ", sk)

    if sk>3 or sk< -3:

      Log_Fare = df[f].map(lambda i: np.log(i) if i > 0 else 0)
      df[f]=Log_Fare
      print("Final skewness in feature: ",f," is: ", Log_Fare.skew())
      fig = px.histogram(Log_Fare, x=f)
      fig.show()
  
  return df



def perform_ordinal_encoding(data, ordinal_features, custom_mapping):

    # Perform ordinal encoding using OrdinalEncoder with the custom mapping
    encoder = OrdinalEncoder(categories=custom_mapping)
    data[ordinal_features] = encoder.fit_transform(data[ordinal_features])

    # Return the encoded DataFrame
    return data


# Fix imbalance in data target using oversampling.

def fix_imbalance_using_oversamping(dataframe, target_column):


  # Separate the features and the target variable
  X = dataframe.drop(target_column, axis=1)
  y = dataframe[target_column]

  # Apply random oversampling using RandomOverSampler
  oversampler = RandomOverSampler(random_state=42)
  X, y = oversampler.fit_resample(X, y)

  return X, y
















