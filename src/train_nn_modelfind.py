"""
Author: Sebastian Burgstaller-Muehlbacher
"""

import tensorflow as tf

from tensorflow import keras
import glob
import os

from resnet import ResnetBuilder

assert os.getenv('LOG_DIR'), 'Logging directory for model saving not set!'
assert os.getenv('TRAIN_DATA_DIR'), 'TRAIN_DATA_PATH not set! This is where training data is read from.'

train_data_path = os.getenv('TRAIN_DATA_DIR')

# Training
batch_size = 40
epochs = 800

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def decode(serialized_example):
    with tf.device('/cpu:0'):
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'train/label': tf.io.FixedLenFeature([], tf.string),
                'train/conv_msa': tf.io.FixedLenFeature([], tf.string),
                'train/tstv_msa': tf.io.FixedLenFeature([], tf.string),
                'pairwise__msa': tf.io.FixedLenFeature([], tf.string),
                'train/seed': tf.io.FixedLenFeature([], tf.int64),
                'train/sec_seed': tf.io.FixedLenFeature([], tf.int64),
                'train/alpha': tf.io.FixedLenFeature([], tf.float32),
                'train/ev_model': tf.io.FixedLenFeature([], tf.string),
            })

        conv_msa = tf.io.decode_raw(features['train/conv_msa'], tf.int8)
        conv_msa = tf.reshape(conv_msa, shape=[40, 2000, 4])

        tstv_msa = tf.io.decode_raw(features['train/tstv_msa'], tf.float16)
        tstv_msa = tf.reshape(tstv_msa, shape=[40, 250, 14])

        pairwise_msa = tf.io.decode_raw(features['pairwise__msa'], tf.float16)
        pairwise_msa = tf.reshape(pairwise_msa, shape=[40, 250, 26])
    
        label = tf.io.decode_raw(features['train/label'], tf.int8)
        label = tf.reshape(label, shape=[12])

        # these 4 lines allow to reduce the 12 models to the 6 actual models and thus spares us geneating training data twice. 
        l = tf.argmax(label, axis=0)
        label = tf.cond(l > 5, lambda: l - 6, lambda: l)
        label = tf.one_hot(label, 12)
        #tf.print(label, summarize=-1)

        # seed = features['train/seed']
        # sec_seed = features['train/sec_seed']
        alpha = features['train/alpha']
        
        alpha = tf.cond(alpha < 0, lambda : tf.constant(50.0), lambda: alpha)
        alpha = tf.math.multiply(alpha, 1000)

    return (pairwise_msa, ),  label


def data_generator(data_files):
    with tf.device('/cpu:0'):
        print(data_files)
        tf_train_data = tf.data.TFRecordDataset(data_files)
        tf_train_data = tf_train_data.map(decode)

        tf_train_data = tf_train_data.repeat()
        tf_train_data = tf_train_data.shuffle(7000)

        # Separate training data into batches
        tf_train_data = tf_train_data.batch(batch_size)
        tf_train_data = tf_train_data.prefetch(2)

        return tf_train_data


print('Build model...')


model = ResnetBuilder.build_resnet_18((26, 40, 250), 12)

adam = keras.optimizers.Adam(lr=0.00001, ) 
model.compile(loss={'dense': 'categorical_crossentropy', },
              optimizer=adam,
              metrics={'dense': 'accuracy',} 
)
model.summary()

print('Loading data...')
train_data_files = glob.glob(os.path.join(train_data_path, '*.tfrecords'))
print('train data files', train_data_files) 

test_data_files = glob.glob(os.path.join(train_data_path, 'test_data', '*.tfrecords'))

print('Train...')

model.fit(data_generator(train_data_files),
          batch_size=None,
          steps_per_epoch=242688 // batch_size,
          epochs=epochs,
          callbacks=[keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}_{val_accuracy:.4f}_1000pos.hdf5',  monitor='val_loss', verbose=0,
                                                    save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1)],
          verbose=2,
          validation_data=(data_generator(test_data_files)),
          validation_steps=12288 // batch_size,
          initial_epoch=0
          )

