import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
import cv2

# import data
dataframe_train = pandas.read_csv('Train.csv', delimiter=';')
dataframe_test = pandas.read_csv('Test.csv', delimiter=';')

dataframe_train['is_train'] = 1
dataframe_test['is_train'] = 0

dataframe = pandas.concat([dataframe_train, dataframe_test])

dataframe = dataframe.reset_index(drop=True)
dataframe.info()

print('Dataframe head')
print(dataframe.head())
print('----------------')

print('Dataframe classid nonunique / unique')
print(dataframe.ClassId.nunique(), dataframe.ClassId.unique())
print('----------------')

# modeling data
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

raw = dataframe[['Filename', 'is_train', 'ClassId']]
raw = pandas.get_dummies(raw, columns=['ClassId'])
raw_train = raw[(raw.is_train == 1)]
raw_test = raw[(raw.is_train == 0)]
X = raw_train.Filename
y = raw_train.iloc[:,2:]

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, shuffle=True)

print('Y valid shape / Y train shape')
print(y_valid.shape, y_train.shape)
print('----------------')

def generator(samples, batch_size=32, is_test= None):
    num_samples = len(samples)

    while 1:
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            labels = []
            for batch_sample in batch_samples:
                local_image = cv2.cvtColor(cv2.imread(batch_sample[0]) , cv2.COLOR_BGR2RGB)
                local_image = cv2.resize(local_image, (50,50))
                images.append(local_image)
                labels.append(batch_sample[1:])

            X_batch = numpy.array(images)
            y_batch = numpy.array(labels)

            if(is_test):
                yield shuffle(X_batch)
            else:
                yield shuffle(X_batch, y_batch)

train_samples = pandas.concat([X_train, y_train], axis=1)
valid_samples = pandas.concat([X_valid, y_valid], axis=1)

train_samples = numpy.array(train_samples)
valid_samples = numpy.array(valid_samples)
batch_size = 4

train_generator = generator(train_samples, batch_size)
valid_generator = generator(valid_samples, batch_size)

raw_test_filtered = numpy.array(raw_test.drop(columns=['is_train']))

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(50,50,3)))
model.add(Conv2D(3,(3,3), activation='relu'))
model.add(Conv2D(8,(3,3), strides=(2,2), activation='relu'))
model.add(Conv2D(32,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64,(5,5), strides=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))

print('Model Summary')
model.summary()
print('-------------')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit_generator(train_generator, epochs=5, verbose=1, steps_per_epoch=len(train_samples)//batch_size, validation_data=valid_generator, validation_steps=len(valid_samples)//batch_size)

model.save('model.h5')

print('Evaluate')
print(model.evaluate_generator(generator(raw_test_filtered, batch_size), steps=len(raw_test_filtered)/batch_size))
print('-------------')