import tensorflow as tf

from tensorflow import keras
import glob
import os
import numpy as np

from tensorflow.keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer
#from keras import initializations
from tensorflow.keras import initializers, regularizers, constraints

assert os.getenv('LOG_DIR'), 'Logging directory for model saving not set!'
assert os.getenv('TRAIN_DATA_DIR'), 'TRAIN_DATA_PATH not set! This is where training data is read from.'

train_data_path = os.getenv('TRAIN_DATA_DIR')


# Convolution
kernel_size = 5
filters = 256
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 40
epochs = 800

block_length = 10000

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
Attention class code modified from: 
https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043/code
"""
class Attention(Layer):
    def __init__(self, step_dim=2498,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def get_config(self):
        return {
            "step_dim": self.step_dim,
            'W_regularizer': self.W_regularizer, 
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias,
            }

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(int(input_shape[-1]),),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = int(input_shape[-1])

        if self.bias:
            print(input_shape)
            self.b = self.add_weight(shape=(int(input_shape[1]),),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = int(self.step_dim)

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim


def decode(serialized_example):
    with tf.device('/cpu:0'):
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'train/label': tf.FixedLenFeature([], tf.string),
                'train/msa': tf.FixedLenFeature([], tf.string),
                'train/seed': tf.FixedLenFeature([], tf.int64),
                'train/sec_seed': tf.FixedLenFeature([], tf.int64),
                'train/alpha': tf.FixedLenFeature([], tf.float32),
                'train/ev_model': tf.FixedLenFeature([], tf.string),
            })

        msa = tf.decode_raw(features['train/msa'], tf.float16)
        msa = tf.reshape(msa, shape=[block_length, 4])
    
        label = tf.decode_raw(features['train/label'], tf.int8)
        l = tf.argmax(label, axis=0)
        label = tf.cond(l > 5, lambda: 1, lambda: 0)
        label = tf.one_hot(label, 12)

        alpha = features['train/alpha']
        
        alpha = tf.cond(alpha < 0, lambda : tf.constant(50.0), lambda: alpha)
        alpha = tf.math.multiply(alpha, 1000)

    return msa, (label, alpha, ) #seed, sec_seed, alpha


def data_generator(data_files):
    with tf.device('/cpu:0'):
        print(data_files)
        tf_train_data = tf.data.TFRecordDataset(data_files)
        tf_train_data = tf_train_data.map(decode)

        tf_train_data = tf_train_data.repeat()
        tf_train_data = tf_train_data.shuffle(120000)

        # Separate training data into batches
        tf_train_data = tf_train_data.batch(batch_size)
        tf_train_data = tf_train_data.prefetch(tf.data.AUTOTUNE)

        return tf_train_data


print('Build model...')

input_data = keras.layers.Input(shape=(10000, 4))

conv1 = keras.layers.Conv1D(filters,
                              kernel_size,
                              padding='valid',
                              activation='relu',
                              strides=1,
                              )(input_data)
conv2 = keras.layers.Conv1D(512, 3, padding='valid', activation='relu', strides=1)(conv1)
conv3 = keras.layers.Conv1D(768, 3, padding='valid', activation='relu', strides=1)(conv2)
pool = keras.layers.MaxPooling1D(pool_size=pool_size)(conv3)
lstm = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(1200, return_sequences=True))(pool)

attention = Attention(2498)(lstm)

out1 = keras.layers.Dense(12, activation='softmax', name='ev_model')(attention)
out2 = keras.layers.Dense(1, activation='linear', name='alpha')(attention)

model = keras.Model(inputs=input_data, outputs=[out1, out2])

adam = keras.optimizers.Adam(lr=0.00001)
model.compile(loss={'ev_model': 'categorical_crossentropy', 'alpha': 'mean_absolute_percentage_error'},
              optimizer=adam,
              metrics={'ev_model': 'accuracy', 'alpha': 'mean_absolute_percentage_error'}
)

model.summary()
print('Loading data...')
train_data_files = glob.glob(os.path.join(train_data_path, '*.tfrecords'))
test_data_files = glob.glob(os.path.join(train_data_path, 'test_data', '*.tfrecords'))

print('Train...')
model.fit(data_generator(train_data_files),
          batch_size=None,
          steps_per_epoch=(242688) // batch_size,
          epochs=epochs,
          callbacks=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',  monitor='val_loss', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1)],
          verbose=2,
          validation_data=(data_generator(test_data_files)),
          validation_steps=12288 // batch_size,
          initial_epoch=0
          )


