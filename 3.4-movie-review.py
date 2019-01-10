from keras.datasets import imdb
from keras import models, layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    """
    Convert list of sequences to one-hot vectors that encode a one for every integer in the sequence
    at that index position.
    :param sequences: list of integers
    :param dimension: int
    :return: ndarray with shape (len(sequences), dimension)
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Load the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# One-hot encode sequences
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convert list to ndarray
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)

# Configure network architecture
model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))  # input vector with size 10000 and output 16
model.add(layers.Dense(16, activation='tanh'))
# model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Final output as a sigmoid probability

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set aside validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train model
history = model.fit(partial_x_train,  # history contains a dict called history.history with loss data
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# plot the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')  # 'bo' is for blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # 'b' is for solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# plot the training and validation accuracy
plt.clf()  # clear figure
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print(results)


