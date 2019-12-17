# use pandas for data manipulation -> provides dataframe as datastructure to store the data
import pandas

# use numpy for data computation -> provides multidimentional array support
import numpy

# use pyplot for data visualization
import matplotlib.pyplot as pyplot

# use tensorflow for deep learning
import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# use for model data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

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


# seperate out features and labels
X = dataframe.drop(columns=['diagnosis'])
y = dataframe.diagnosis

sc = StandardScaler()
X = sc.fit_transform(X)

# since machines understand language of either 0 or 1, you have to provide them data in that language only.
# so convert M to 0 and B to 1

le = LabelEncoder()
y = le.fit_transform(y)

# set aside Training and test data for validation of our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state = 42)

# checking out shape of the variables for traiing and test
print('shape of the variables for traiing and test')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('--------------')

model = Sequential()

model.add(Dense(32, input_shape=(31,)))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy' , metrics=['accuracy'], optimizer=Adam(lr=0.0001))

model.fit(X_train, y_train, validation_split = 0.1, epochs= 50, verbose =1, batch_size = 8)

# we perform prediction on the validation set kept aside in step 4
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)

# for validation set
confusion_matrix( y_test, y_pred)

pyplot.show()