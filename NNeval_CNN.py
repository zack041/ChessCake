import chess
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def encode_board(board):
    matrix = np.zeros((8,8,13), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            matrix[i//8][i%8][piece.piece_type-1+(0 if piece.color ^ board.turn else 6)] = 1
        else:
            matrix[i//8][i%8][12] = 1
    return matrix

df = pd.read_csv('train.csv')
df = df[~df['Evaluation'].astype(str).str.startswith('#')].copy()
df['Evaluation'] = df['Evaluation'].astype(int)
df = df.reset_index(drop=True)
test_df= pd.read_csv('test.csv')

def input_proc(df):
    X = []
    y = []
    for i in tqdm(range(len(df['FEN']))):
        bd = chess.Board(df['FEN'][i])
        X.append(encode_board(bd))
        y.append(df['Evaluation'][i] * (1 if bd.turn else -1))
    return np.array(X, dtype = 'float32'), np.array(y, dtype = 'float32')

X_train, y_train = input_proc(df)
X_test, y_test = input_proc(test_df)

model = keras.Sequential([
    keras.Input(shape=(8,8,13)),
    layers.Conv2D(32, (3, 3), activation='tanh', padding='same'),
    layers.Conv2D(64, (3, 3), activation='tanh', padding='same'),
    layers.Conv2D(128, (5, 5), activation='tanh', padding='same'),
    layers.Flatten(),
    layers.Dense(256, activation='tanh'),
    layers.Dense(1)
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_absolute_error',
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=512,
    validation_data=(X_test, y_test),
)

model.save('NNeval_CNN.keras')
