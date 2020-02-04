import csv
import os
from matplotlib import pyplot as plt
from pprint import pprint
import plotting

import numpy as np
from numba import njit

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with open('jena_climate_2009_2016.csv') as f:
    lines = []
    datareader = csv.reader(f)
    for row in datareader:
        lines.append(row)

header = lines[0]
lines.pop(0)

print(header)

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    float_data[i] = line[1:]


temp = float_data[:, 1]
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# def generator(data, lookback, delay, min_index, max_index,
#               shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             i += len(rows)
#
#         samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + delay][1]
#
#         yield samples, targets


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples, targets = gen_loop(rows, data, delay, lookback, step)

        yield samples, targets

@njit
def gen_loop(rows, data, delay, lookback, step):
    samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
    targets = np.zeros((len(rows),))
    for j in range(len(rows)):
        indices = np.arange(rows[j] - lookback, rows[j], step)
        samples[j] = data[indices]
        targets[j] = data[rows[j] + delay][1]

    return samples, targets


lookback = 1440  # 10 days
step = 6  # 1 hour steps
delay = 144  # 24 hour period
batch_size = 128
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)

test_steps = (len(float_data) - 300001 - lookback)


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
        if step % 1000 == 0:
            print(step, mae)
    print(np.mean(batch_maes))

# print(evaluate_naive_method())
# celsius_mae = 0.29 * std[1]
# print(celsius_mae)

# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop

# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# plotting.plot_loss(history, '6.20_loss.png')

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

plotting.plot_loss(history, '6.21_loss.png')