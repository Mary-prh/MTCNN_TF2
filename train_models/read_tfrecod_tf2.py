import tensorflow as tf

def image_color_distort(inputs):
    """Applies random adjustments to brightness, contrast, hue, and saturation."""
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
    # inputs = tf.image.random_flip_left_right(inputs, seed=5)
    return inputs


def random_flip_images(image_batch, label_batch, roi_batch,landmark_batch):
    # Random condition for flipping each image in the batch
    flip_cond = tf.random.uniform([tf.shape(image_batch)[0]], 0, 1) > 0.5
    
    # Find indexes for landmarks and positives that may be flipped
    flip_landmark_indexes = tf.where((label_batch == -2) & flip_cond)
    flip_pos_indexes = tf.where((label_batch == 1) & flip_cond)

    # Consolidate all indexes that should be considered for flipping
    flip_indexes = tf.concat([flip_landmark_indexes, flip_pos_indexes], axis=0)

    # Conditionally flip images
    def flip_image(idx):
        # Only flip if the index is in the flip indexes
        return tf.image.flip_left_right(image_batch[idx])
    
    # Map function to apply flipping based on the condition
    image_batch = tf.map_fn(lambda idx: tf.cond(flip_cond[idx], 
                                                lambda: flip_image(idx), 
                                                lambda: image_batch[idx]), 
                            tf.range(tf.shape(image_batch)[0], dtype=tf.int64), dtype=tf.float32)

    # Flip landmarks corresponding to the flipped images, if they are landmarks
    def flip_landmarks(landmarks):
        landmarks = tf.reshape(landmarks, (-1, 2))
        landmarks_flipped = tf.stack([1 - landmarks[:, 0], landmarks[:, 1]], axis=1)
        # Swap left and right points, assuming indices 0-1 are eyes and 3-4 are mouth corners
        landmarks_flipped = tf.gather(landmarks_flipped, [1, 0, 2, 4, 3], axis=0)
        return tf.reshape(landmarks_flipped, [-1])

    landmark_batch = tf.map_fn(lambda idx: tf.cond(flip_cond[idx] & tf.reduce_any(tf.equal(flip_indexes[:, 0], idx)), 
                                                   lambda: flip_landmarks(landmark_batch[idx]), 
                                                   lambda: landmark_batch[idx]), 
                               tf.range(tf.shape(landmark_batch)[0], dtype=tf.int64), dtype=tf.float32)

    return image_batch, label_batch, roi_batch, landmark_batch

def read_single_tfrecord(tfrecord_file, batch_size, net):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/roi': tf.io.FixedLenFeature([4], tf.float32),
        'image/landmark': tf.io.FixedLenFeature([10], tf.float32)
    }
    # image size based on the net
    image_size = {'PNet': 12, 'RNet': 24, 'ONet': 48}.get(net, 12)
    # Parse a single example
    def parse_example(serialized_example):
        parsed_record = tf.io.parse_single_example(serialized_example, features)
        image = tf.io.decode_raw(parsed_record['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32) - 127.5) / 128
        image = image_color_distort(image)
        label = tf.cast(parsed_record['image/label'], tf.float32)
        roi = tf.cast(parsed_record['image/roi'], tf.float32)
        landmark = tf.cast(parsed_record['image/landmark'], tf.float32)
        return image, label, roi, landmark

    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda images, labels, rois, landmarks: (random_flip_images(images, labels, rois, landmarks)), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset