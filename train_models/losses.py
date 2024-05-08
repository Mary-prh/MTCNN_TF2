import tensorflow as tf

# OHEM Functions
num_keep_radio = 0.7

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    # pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # get the number of rows of class_prob
    num_row = tf.cast(cls_prob.get_shape()[0], tf.int32)
    # row = [0,2,4.....]
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.math.log(label_prob + 1e-10)

    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros, zeros, ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

def bbox_ohem(bbox_pred, bbox_target, label):
    """
    Perform Online Hard Example Mining (OHEM) for bounding box regression loss.
    :param bbox_pred: Predicted bounding box offsets (shape: [batch_size, 4]).
    :param bbox_target: Ground truth bounding box offsets (shape: [batch_size, 4]).
    :param label: Ground truth labels (shape: [batch_size]).
    :return: Mean bounding box regression loss.
    """
    # Mask to filter positive (1) and part (-1) examples only
    bbox_pred = tf.reshape(bbox_pred, [-1, 4])
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), 1.0, 0.0)

    # Ensure valid_inds is a 1D tensor
    valid_inds = tf.reshape(valid_inds, [-1])

    # Calculate the sum of squared differences for each predicted bbox
    square_error = tf.reduce_sum(tf.square(bbox_pred - bbox_target), axis=1)

    # Mask out invalid examples
    square_error = square_error * valid_inds

    # Count the number of positive and part examples
    num_valid = tf.reduce_sum(valid_inds)

    # Determine the number of top errors to keep
    keep_num = tf.cast(num_valid, dtype=tf.int32)

    # Find top-k errors
    _, k_index = tf.nn.top_k(square_error, k=keep_num)

    # Gather the top errors
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


# def landmark_ohem(landmark_pred, landmark_target, label):
#     # Mask to keep only landmarks with label -2
#     landmark_pred = tf.reshape(landmark_pred, [-1, 10])
#     # print("landmark_pred, landmark_target shapes:" , landmark_pred.shape, landmark_target.shape)
#     mask_landmark = tf.equal(label, -2)

#     # Apply mask
#     valid_landmark_pred = tf.boolean_mask(landmark_pred, mask_landmark)
#     valid_landmark_target = tf.boolean_mask(landmark_target, mask_landmark)

#     if tf.size(valid_landmark_pred) == 0:
#         return tf.constant(0.0, dtype=tf.float32)

#     square_error = tf.square(valid_landmark_pred - valid_landmark_target)
#     square_error = tf.reduce_sum(square_error, axis=1)

#     keep_num = tf.cast(tf.shape(valid_landmark_pred)[0], dtype=tf.int32)
#     _, k_index = tf.nn.top_k(square_error, k=keep_num)
#     square_error = tf.gather(square_error, k_index)

#     return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred, landmark_target, label):
    """Calculate landmark loss with Online Hard Example Mining (OHEM)."""
    # Mask to keep only landmarks with label -2
    landmark_pred = tf.reshape(landmark_pred, [-1, 10])
    mask_landmark = tf.equal(label, -2)

    # Apply mask
    valid_landmark_pred = tf.boolean_mask(landmark_pred, mask_landmark)
    valid_landmark_target = tf.boolean_mask(landmark_target, mask_landmark)

    # Count the number of valid landmarks
    num_valid_landmarks = tf.shape(valid_landmark_pred)[0]

    def compute_loss():
        square_error = tf.square(valid_landmark_pred - valid_landmark_target)
        square_error = tf.reduce_sum(square_error, axis=1)

        keep_num = tf.cast(tf.shape(valid_landmark_pred)[0], dtype=tf.int32)
        _, k_index = tf.nn.top_k(square_error, k=keep_num)
        square_error = tf.gather(square_error, k_index)

        return tf.reduce_mean(square_error)

    def return_zero_loss():
        return tf.constant(0.0, dtype=tf.float32)

    # Conditionally compute loss based on the number of valid landmarks
    return tf.cond(num_valid_landmarks > 0, compute_loss, return_zero_loss)

