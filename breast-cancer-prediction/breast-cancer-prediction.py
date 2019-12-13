# Use pandas for data manipulation -> provides dataframe as datastructure to store the data
import pandas

# use numpy for data computation -> provides multidimentional array support
import numpy

# use pyplot for data visualization
import matplotlib.pyplot as pyplot

#use tensorflow for deep learning
import tensorflow

# read data from csv file
dataframe = pandas.read_csv('./data.csv')

# print first 5 rows from the data csv
print('DataFrame Head')
print(dataframe.head())
print('--------------')

# print a descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values
print('DataFrame Descriptive Statistics')
print(dataframe.describe())
print('--------------')

# print a concise summary of a DataFrame
print('DataFrame Info')
dataframe.info()
print('--------------')

#checking if any column has a null value because nulls can cause a problem while training model
dataframe.isnull().sum()

# remove unsed column
dataframe.drop(columns=['Unnamed: 32'], inplace=True)

malignant = dataframe[dataframe.diagnosis == 'M']
benign = dataframe[dataframe.diagnosis == 'B']

fig, ax = pyplot.subplots(figsize =(16,4))
pyplot.plot(malignant.concavity_mean, label = 'Malignant')
pyplot.plot(benign.concavity_mean, label = 'Benign')
pyplot.legend()

pyplot.show()