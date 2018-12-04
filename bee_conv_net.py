import tensorflow as tf


def bee_conv_net_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 48, 48, 3])

    # Batch norm layer
    bnorm = tf.layers.batch_normalization(inputs=input_layer)

    # Convolutional Layer 1: (?, 5, 5, 3, 8) -> (?, 24, 24, 8)
    conv1 = tf.layers.conv2d(
        inputs=bnorm,
        filters=8,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        strides=2)

    # Convolutional Layer 2: (?, 5, 5, 8, 16) -> (?, 12, 12, 16)
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=16,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        strides=2)

    # Dense layer
    flat = tf.reshape(conv2, [-1, 12 * 12 * 16])
    dense = tf.layers.dense(inputs=flat, units=350, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.6,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    accuracy = tf.metrics.accuracy(labels, predictions['classes'], name='accuracy')

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(0.005, tf.train.get_global_step(), 1000, 0.96, name='learning_rate')
        logging_hook = tf.train.LoggingTensorHook({
            "loss": loss,
            "accuracy": accuracy[1],
            "learning_rate": lr
        }, every_n_iter=100)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    eval_metric_ops = {'accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
