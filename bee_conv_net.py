import tensorflow as tf


def bee_conv_net_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 200, 200, 3])

    # Batch norm layer
    bnorm = tf.layers.batch_normalization(inputs=input_layer)

    # Convolutional Layer 1: (?, 5, 5, 3, 4) -> (?, 100, 100, 4)
    conv1 = tf.layers.conv2d(
        inputs=bnorm,
        filters=4,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        strides=2)

    # Convolutional Layer 2: (?, 6, 6, 4, 6) -> (?, 50, 50, 6)
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=6,
        kernel_size=[6, 6],
        padding='same',
        activation=tf.nn.relu,
        strides=2)

    # Convolutional Layer 3: (?, 8, 8, 6, 8) -> (?, 10, 10, 8)
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=8,
        kernel_size=[8, 8],
        padding='same',
        activation=tf.nn.relu,
        strides=5)

    # Dense layer
    conv3_flat = tf.reshape(conv3, [-1, 800])
    dense = tf.layers.dense(inputs=conv3_flat, units=200, activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=2)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predictions['classes']),
        'recall': tf.metrics.recall(labels, predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
