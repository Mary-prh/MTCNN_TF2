import tensorflow as tf

def ohem_loss(labels, logits, num_keep_radio=0.7):
    """ OHEM Loss for batch of samples
    Args:
    labels: true labels, shape = (batch_size,)
    logits: logits from the model, shape = (batch_size, num_classes)
    num_keep_radio: the ratio of examples to keep based on the loss

    Returns:
    loss: scalar tensor of the OHEM loss
    """
    # Perform softmax on logits to compute probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    # Get true class probabilities based on label index
    labels_int = tf.cast(labels, tf.int32)
    true_probs = tf.gather(probs, labels_int, axis=1, batch_dims=1)

    # Compute the loss (negative log likelihood)
    ohem_loss = -tf.math.log(true_probs + 1e-10)

    # Filter out the invalid labels (-1 and -2 should be treated as ignore)
    valid_mask = labels >= 0
    valid_loss = tf.boolean_mask(ohem_loss, valid_mask)

    # Compute number of valid examples to keep
    num_valid = tf.cast(tf.size(valid_loss), tf.float32)
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    
    # Only keep top-k loss values
    topk_loss, _ = tf.nn.top_k(valid_loss, k=keep_num)

    # Average the top-k loss values
    return tf.reduce_mean(topk_loss)

def bbox_ohem_loss(y_true, y_pred, label):
    """
    Online Hard Example Mining Loss for bounding box regression.

    Args:
    y_true: Tensor of true bounding box coordinates (batch_size, 4).
    y_pred: Tensor of predicted bounding box coordinates (batch_size, 4).
    label: Tensor of class labels (batch_size,).

    Returns:
    Scalar Tensor representing the mean loss for the hard examples.
    """
    # Convert label to float32 to match y_pred's dtype
    label = tf.cast(label, tf.float32)
    
    # Mask to select positive and part examples (where label == 1)
    mask = tf.equal(tf.abs(label), 1)
    mask = tf.reshape(mask, [-1, 1])  # Reshape mask to [batch_size, 1] to broadcast
    mask = tf.tile(mask, [1, 4])  # Tile mask to match bbox dimensions [batch_size, 4]

    # Calculate squared error only for selected examples
    squared_error = tf.square(y_pred - y_true)
    squared_error = tf.reduce_sum(squared_error, axis=1)  # Sum across coordinates

    # Apply mask
    masked_squared_error = tf.where(mask, squared_error, tf.zeros_like(squared_error))

    # Find the number of valid (non-zero) errors to determine the top k hardest examples
    num_valid = tf.reduce_sum(tf.cast(mask, tf.int32), axis=0)

    # Use top_k to select the hard examples
    values, indices = tf.nn.top_k(masked_squared_error, k=num_valid[0])

    # Compute the mean loss over these hardest examples
    return tf.reduce_mean(values)

def landmark_ohem_loss(y_true, y_pred, labels):
    """
    Calculates the Online Hard Example Mining Loss for landmarks.
    
    Args:
    y_true: true landmark positions (batch_size, 10)
    y_pred: predicted landmark positions (batch_size, 10)
    labels: class labels indicating which samples are for landmark detection (batch_size,)

    Returns:
    Scalar tensor of the OHEM loss for landmarks.
    """
    # Identify valid examples (those meant for landmark detection)
    mask = tf.equal(labels, -2)
    
    # Compute squared error for valid examples
    squared_error = tf.square(y_pred - y_true)
    squared_error = tf.reduce_sum(squared_error, axis=1)  # Sum across all landmark dimensions

    # Apply the mask to focus loss calculation on valid examples
    masked_squared_error = tf.where(mask, squared_error, tf.zeros_like(squared_error))

    # Count valid examples and determine the number to keep
    num_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
    topk_values, _ = tf.nn.top_k(masked_squared_error, k=num_valid)

    # Calculate the mean loss over the top hardest examples
    return tf.reduce_mean(topk_values)
