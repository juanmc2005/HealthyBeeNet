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
    conv3_flat = tf.reshape(conv2, [-1, 12 * 12 * 16])
    dense = tf.layers.dense(inputs=conv3_flat, units=500, activation=tf.nn.relu)

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

    y_pred = predictions['classes']

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, y_pred),
        'recall': tf.metrics.recall(labels, y_pred)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
