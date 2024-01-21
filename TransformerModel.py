import numpy as np
import pandas as pd
import logging
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Attention, Bidirectional, Input, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.python.client import device_lib
from keras.regularizers import l2
from keras.layers import Reshape
from keras.models import Model
from keras.layers import MultiHeadAttention, LayerNormalization
from keras.callbacks import LearningRateScheduler
import tensorflow as tf



class TransformerModel():
    def __init__(self, df, features, predictions) -> None:
        self.df = df
        self.features = features
        self.predictions = predictions
        self.set_up_logger()
        self.x_sequences = []
        self.y_sequences = []
        self.max_position = 0
        self.model = None

    def set_up_logger(self):
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)


    def check_for_gpu(self):
        if 'GPU' in str(device_lib.list_local_devices()): return True 
        else: return False


    def split_into_sequences(self):
        print(f"features: {self.features}")
        print(f"predictions: {self.predictions}")
        print(f"preiction loc: {self.df.columns.get_loc(self.predictions)}")

        # Prepare for conversion to sequences
        sequence_length = 56 # 8 readings per day * 7 days
        self.max_position = 56
        X_sequences = []
        y_sequences = []

        # Extract the features into sequences
        for i in range(len(self.df)):
            end_ix = i + sequence_length
            if end_ix > len(self.df) - 1:
                break

            # Extract the input sequence (features) and the corresponding target value
            seq_x = self.df.iloc[i:end_ix, 1:]  # Exclude timestamp from X
            seq_y = self.df.iloc[end_ix, 2]

            # Append the sequences to X and y
            X_sequences.append(seq_x.values)
            y_sequences.append(seq_y)

        # Convert lists to NumPy arrays
        self. x_sequences = np.array(X_sequences)
        self.y_sequences = np.array(y_sequences)

        # print shapes of the arrays
        # X seq shape = (num_samples, sequence_length, num_features)
        print(f"X_sequences shape: {self.x_sequences.shape}")
        print(f"y_sequences shape: {self.y_sequences.shape}")


    def createTransformerModel(self):
        # Specify the number of time steps and features
        n_steps, n_features = self.x_sequences.shape[1], self.x_sequences.shape[2]
        inputs = Input(shape=(n_steps, n_features))

        # Positional Encoding
        positions = tf.range(start=0, limit=self.max_position, delta=1)
        position_embeddings = 2 * (positions[:, tf.newaxis] / np.float16(self.max_position)) - 1
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        position_embeddings = tf.keras.layers.Embedding(input_dim=self.max_position, output_dim=n_features)(positions)
        x = inputs + position_embeddings

        # Transformer Layers with Normalization
        for _ in range(6):
            transformer_out = MultiHeadAttention(
                num_heads=64, 
                key_dim=512, 
                dropout=0.1,
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                bias_regularizer=l2(0.01),
                kernel_regularizer=l2(0.01)
            )(x, x, x)
            norm_attention = LayerNormalization(epsilon=1e-6)(transformer_out)
            x = Concatenate()([x, norm_attention])

        # Temporal Convolutional Layers
        for _ in range(3):
            conv1d_out = Conv1D(filters=1024, kernel_size=1, activation='relu', kernel_regularizer=l2(0.01))(x)
            maxpool_out = MaxPooling1D(pool_size=2)(conv1d_out)
            x = maxpool_out

        # Output layer
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        # Ouput is the number of predictions we want.
        # Right now we are predicting 1 value, air_temp
        output_layer = Dense(len(self.predictions))(x)

        self.model = Model(inputs=inputs, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mse') 


    def trainModel(self):
        if self.check_for_gpu():
            def learning_rate_schedule(epoch):
                initial_learning_rate = 0.001
                decay = 0.025
                lr = initial_learning_rate / (1 + decay * epoch)
                return lr
            lr_scheduluer = LearningRateScheduler(learning_rate_schedule)

            self.model.fit(
                self.x_sequences, 
                self.y_sequences, 
                epochs=75, 
                batch_size=32, 
                callbacks=[lr_scheduluer]
                )
        else:
            print("GPU not recognized")


    def saveModel(self):
        self.model.save("saved_Models/TransfromerModel.keras")




