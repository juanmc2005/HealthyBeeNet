import bee_data as bees
import image as imgs
import evaluation as evl
import bee_conv_net as bcn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Random Seed
tf.random.set_random_seed(1)

# Logging
tf.logging.set_verbosity(tf.logging.INFO)

img_size = 48
dataset_dir = 'bee_dataset/'
bee_img_dir = dataset_dir + 'bee_imgs/'

print("Loading dataset metadata")

# Load dataset
train_x_files = bees.read_x_split(dataset_dir + 'train_x')
train_y = np.asarray(bees.read_y_split_binary(dataset_dir + 'train_y'))
test_x_files = bees.read_x_split(dataset_dir + 'test_x')
test_y = np.asarray(bees.read_y_split_binary(dataset_dir + 'test_y'))

train_x_files, val_x_files, train_y, val_y = train_test_split(
    train_x_files, train_y, test_size=0.1, stratify=train_y, random_state=1)

print("Training Set Size: {}".format(len(train_x_files)))
print("Validation Set Size: {}".format(len(val_x_files)))
print("Test Set Size: {}".format(len(test_x_files)))
print("Loading dataset images")

# 4133 RGB images of (392x520). Tensor has shape (?, img_size, img_size, 3)
train_x = np.asarray(imgs.load_images_fit_size(
    bee_img_dir, train_x_files, width=img_size, height=img_size), dtype=np.float32)
val_x = np.asarray(imgs.load_images_fit_size(
    bee_img_dir, val_x_files, width=img_size, height=img_size), dtype=np.float32)
test_x = np.asarray(imgs.load_images_fit_size(
    bee_img_dir, test_x_files, width=img_size, height=img_size), dtype=np.float32)

print("Dataset loaded")

# Build the classifier
bee_classifier = tf.estimator.Estimator(
    model_fn=bcn.bee_conv_net_fn,
    model_dir='/tmp/bee_model_test')

print("Classifier built")

# Create the training input function (params for training)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_x},
    y=train_y,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

print("Will now train")

# Train the classifier
# bee_classifier.train(input_fn=train_input_fn, steps=20000)

print("Training finished")

# Create the evaluation input function (params for evaluation)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_x},
    y=test_y,
    num_epochs=1,
    shuffle=False)

# Evaluate the classifier
eval_results = bee_classifier.evaluate(input_fn=eval_input_fn)
pred_results = bee_classifier.predict(eval_input_fn)

print('Global Accuracy: {}'.format(eval_results['accuracy']))

pred_y = [bool(y['classes']) for y in pred_results]

print('Results in Validation Set:')
print(evl.class_precision_recall(test_y, pred_y))
