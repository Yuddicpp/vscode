# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import *

# define settings
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_classes', 50, 'number of classes used')
flags.DEFINE_integer('num_samples_train', 15, 'number of samples per class used for training')
flags.DEFINE_integer('num_samples_test', 5, 'number of samples per class used for testing')
flags.DEFINE_integer('seed', 1, 'random seed')

# define you model, loss functions, hyperparameters, and optimizers
### Your Code Here ###



# load data
train_image, train_label, test_image, test_label = LoadData(FLAGS.num_classes, FLAGS.num_samples_train, FLAGS.num_samples_test, FLAGS.seed)
# note: you should use train_image, train_label for training, apply the model to test_image to get predictions and use test_label to evaluate the predictions 
train_image = train_image.reshape(-1,28,28)
test_image = test_image.reshape(-1,28,28)

with tf.Session() as sess:
    # initialize 
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    # train model using train_image and train_label
    ### Your Code Here ###
    
    
    
    # get predictions on test_image
    ### Your Code Here ###
    pred = None
    
    
    # evaluation
    print("Test Accuracy:", np.mean(1.0 * (pred == test_label)))
    # note that you should not use test_label elsewhere





