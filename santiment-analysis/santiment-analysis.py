import tensorflow
from tensorflow import keras
import matplotlib.pyplot as pyplot
import numpy
import seaborn

ret = keras.datasets.reuters
(X_train, y_train), (X_test, y_test) = ret.load_data(num_words=5000)

#for checking the unique labels
numpy.unique(y_train)

print('Number of words: ')
print(len(numpy.unique(numpy.hstack(X_train))))
print('------------------')

# Visualizing the Dataset
result = [len(x) for x in X_train]
seaborn.boxplot(y=result)

print('Mean is')
print(numpy.mean(result))
print('------------------')

word_index = ret.get_word_index()
reverse_word_index = {y:x for x,y in word_index.items()}

def decode_review(encoded_review):
    decoded_review = []
    for word in encoded_review:
        if word in reverse_word_index:
            decoded_review.append(reverse_word_index[word])
    return decoded_review

decode_review(X_train[0])

word_index = {x:(y+3) for x,y in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
reverse_word_index = {y:x for x,y in word_index.items()}

print(' '.join(decode_review(X_train[0])))
print('----------------')
print(' '.join(decode_review(X_train[3])))

# Padding all Reviews to have equal length
from tensorflow.python.keras.preprocessing import sequence

max_review_length = 500
X_train_padded = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)

result = [len(x) for x in X_train_padded]
seaborn.boxplot(y=result)

# One Hot Encoding
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
token_X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
token_X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

# Modelling Data
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam

embedding_vector_length = 512
model = Sequential()
model.add(Embedding(5000, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100, return_sequences=True))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(46, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.1)
model.summary()

model.save_weights('sentiment.h5')

val_loss = model.history.history['val_loss']
tra_loss = model.history.history['loss']

pyplot.plot(val_loss)
pyplot.plot(tra_loss)
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.title('Loss Curve')
pyplot.legend(['Validation Loss', 'Training Loss'])
pyplot.show()

pyplot.show()
