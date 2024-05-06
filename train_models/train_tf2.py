import os
import math
import tensorflow as tf
from MTCNN_config import config
from read_tfrecod_tf2 import read_single_tfrecord
from mtcnn import create_pnet
from losses import *
from keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay 

# >>>>>>>>>>> solution #3
def get_model_config(net_type):
    if net_type == 'PNet':
        image_size = 12
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 0.5}
    elif net_type == 'RNet':
        image_size = 24
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 0.5}
    else:  # ONet
        image_size = 48
        loss_weights = {'classifier': 1.0, 'bbox_regress': 0.5, 'landmark_pred': 1.0}
    return image_size, loss_weights


def get_lr_schedule(base_lr, data_num, batch_size, lr_epochs):
    boundaries = [int(epoch * data_num / batch_size) for epoch in lr_epochs]
    values = [base_lr * (0.1 ** i) for i in range(len(boundaries) + 1)]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return lr_schedule

# Training loop
def train_step(model, images, labels, rois, landmarks, classification_loss, bbox_loss, landmark_loss,optimizer):
    # Optimizer
    with tf.GradientTape() as tape:
        classifier_output, bbox_output, landmark_output = model(images, training=True)

        # Mask for classification (only negatives and positives)
        mask_cls = tf.logical_or(tf.equal(labels, 1), tf.equal(labels, 0))
        # loss_classifier = classification_loss(tf.boolean_mask(labels, mask_cls), 
        #                                       tf.boolean_mask(classifier_output, mask_cls))* loss_weights['classifier']
        
        # OHEM: Select hard examples (top 70% of losses)
        loss_classifier_full = classification_loss(tf.boolean_mask(labels, mask_cls), 
                                              tf.boolean_mask(classifier_output, mask_cls))
        loss_classifier_full_flat = tf.reshape(loss_classifier_full, [-1])
        num_samples = tf.size(loss_classifier_full_flat, out_type=tf.float32)  # Ensure the size is float for multiplication
        num_hard_examples = tf.cast(0.7 * num_samples, tf.int32)
        if num_hard_examples > 0:
            values, indices = tf.nn.top_k(loss_classifier_full_flat, k=num_hard_examples, sorted=True)
            # Use only the selected hard examples for calculating the mean classification loss
            loss_classifier = tf.reduce_mean(tf.gather(loss_classifier_full_flat, indices)) * loss_weights['classifier']
        else:
            loss_classifier = 0
        
        loss_classifier += 1e-6 * tf.reduce_sum(classifier_output)
        # Mask for bounding box regression (only positives and part faces)
        mask_bbox = tf.logical_or(tf.equal(labels, 1), tf.equal(labels, -1))
        if tf.reduce_any(mask_bbox):
            loss_bbox = bbox_loss(tf.boolean_mask(rois, mask_bbox), 
                                tf.boolean_mask(bbox_output, mask_bbox)) * loss_weights['bbox_regress']
        else:
            loss_bbox = 0
        loss_bbox += 1e-6 * tf.reduce_sum(bbox_output)
        
        # Mask for landmark localization (only landmark faces)
        mask_landmark = tf.equal(labels, -2)
        if tf.reduce_any(mask_landmark):
            loss_landmark = landmark_loss(tf.boolean_mask(landmarks, mask_landmark), 
                                        tf.boolean_mask(landmark_output, mask_landmark)) * loss_weights['landmark_pred']
        else:
            loss_landmark = 0
        loss_landmark += 1e-6 * tf.reduce_sum(landmark_output)

        total_loss = (loss_classifier + loss_bbox + loss_landmark)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, loss_classifier, loss_bbox, loss_landmark

def print_progress(iteration, total, loss, prefix='', suffix='', decimals=1, length=50, fill='>'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # print(f'\r{prefix} |{bar}| {percent}% Loss: {loss:.4f} {suffix}', end='\r')
    print(f'\r{prefix} |{bar}| {percent}% [{iteration}/{total}] Loss: {loss:.4f} {suffix}', end='\r')
    if iteration == total:
        print()

# Load dataset
base_dir = './DATA/imglists/PNet'
dataset_dir = os.path.join(base_dir,f'train_PNet_landmark.tfrecord_shuffle')
print('dataset dir is:',dataset_dir)
batch_size = config.BATCH_SIZE
dataset = read_single_tfrecord(dataset_dir,batch_size , "PNet")

net = 'PNet'
image_size      = get_model_config(net_type=net)[0]
loss_weights    = get_model_config(net_type=net)[1]
print(image_size)
pnet = create_pnet(image_size,image_size)
num_epochs = 5


# Loss functions
classification_loss = SparseCategoricalCrossentropy(from_logits=False)
bbox_loss = MeanSquaredError()
landmark_loss = MeanSquaredError()

# Calculate expected batches and compare with actual count
data_dir = os.path.join(base_dir,'train_PNet_landmark.txt')
f = open(data_dir, 'r')
num_samples = len(f.readlines())
num_batches = math.ceil(num_samples / batch_size)

# optimizer
epochs = [2, 14]
base_lr = 0.01
lr_factor = 0.1
learning_rates = [base_lr * (lr_factor ** i) for i in range(len(epochs) + 1)]

steps_per_epoch = num_samples // config.BATCH_SIZE
boundaries = [step * steps_per_epoch for step in epochs]
learning_rate_schedule = PiecewiseConstantDecay(boundaries, learning_rates)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule, momentum=0.9)

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    epoch_loss = 0
    num_processed_batches = 0
    for i, (images, labels, rois, landmarks) in enumerate(dataset):
        loss_values, _, _, _ = train_step(model=pnet,images=images, labels=labels, rois=rois, landmarks=landmarks, 
                                 classification_loss=classification_loss, bbox_loss=bbox_loss,landmark_loss=landmark_loss,
                                 optimizer=optimizer)  
        current_loss = loss_values
        epoch_loss += current_loss
        num_processed_batches += 1
        # Update the progress bar with the correct loss value
        print_progress(i + 1, num_batches, current_loss, prefix=i, suffix='Complete', length=40)

        # Check if the batch count exceeds the expected number of batches
        if num_processed_batches >= num_batches:
            break
    average_epoch_loss = epoch_loss / num_processed_batches
    print(f'/nEpoch {epoch+1}, average_epoch_loss: {average_epoch_loss:.4f}')

print("Training complete!")

# >>>>>>>>>>> solution #2

# def get_lr_schedule(base_lr, data_num, batch_size, lr_epochs):
#     boundaries = [int(epoch * data_num / batch_size) for epoch in lr_epochs]
#     values = [base_lr * (0.1 ** i) for i in range(len(boundaries) + 1)]
#     lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
#     return lr_schedule

# def compile_model(optimizer):
#     width, height = 12, 12  # Adjust based on your input image size requirements
#     model, loss_weights = create_pnet(width, height)
#     base_lr = 0.01
#     data_num = 50000  # Update with your actual dataset size
#     batch_size = 128
#     lr_epochs = config.LR_EPOCH
#     lr_schedule = get_lr_schedule(base_lr, data_num, batch_size, lr_epochs)

#     model.compile(
#         optimizer=optimizer,
#         loss={
#             'conv4-1': ohem_loss,
#             'conv4-2': bbox_ohem_loss,
#             'conv4-3': landmark_ohem_loss
#         },
#         loss_weights=loss_weights,
#         metrics={'conv4-1': 'accuracy'} 
#     )
#     return model

# @tf.function
# def train_step(inputs, bbox_targets, landmark_targets, labels, model, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         classifier_output, bbox_output, landmark_output = predictions

#         # Apply OHEM losses
#         cls_loss = ohem_loss(labels, classifier_output)
#         bbox_loss = bbox_ohem_loss(bbox_targets, bbox_output, labels)
#         landmark_loss = landmark_ohem_loss(landmark_targets, landmark_output, labels)

#         # Combine losses
#         total_loss = cls_loss + bbox_loss + landmark_loss

#     gradients = tape.gradient(total_loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return total_loss


# def main():
#     base_lr = 0.01
#     data_num = 50000  # Update with your actual dataset size
#     batch_size = 128
#     lr_schedule = get_lr_schedule(base_lr, data_num, batch_size, config.LR_EPOCH)
#     optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
#     model = compile_model(optimizer)
#     base_dir = './DATA/imglists/PNet'
#     dataset_dir = os.path.join(base_dir,f'train_PNet_landmark.tfrecord_shuffle')
#     print('dataset dir is:',dataset_dir)
#     dataset = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, "PNet")
#     epochs = 10 

#     for epoch in range(epochs):
#         for images, labels, bbox_targets, landmark_targets in dataset:
#             loss = train_step(images, bbox_targets, landmark_targets, labels, model, optimizer)
#             print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# if __name__ == "__main__":
#     main()


# >>>>>>>>>>> solution #1
# def train(net_factory, prefix, base_dir,
#           display=200, base_lr=0.01, epochs = 10):
#     net = os.path.basename(prefix)
#     label_file = os.path.join(base_dir,f'train_{net}_landmark.txt')
#     f = open(label_file, 'r')
#     num = len(f.readlines())
#     print("Total size of the dataset is: ", num)

#     if net == 'PNet':
#         dataset_dir = os.path.join(base_dir,f'train_{net}_landmark.tfrecord_shuffle')
#         print('dataset dir is:',dataset_dir)
#         dataset = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
#         # print("Images shape:", image_batch.shape)
#         # print("Labels shape:", label_batch.shape)
#         # print("ROIs shape:", bbox_batch.shape)
#         # print("Landmarks shape:", landmark_batch.shape)
#     else:
#         print("edit code here!")
#         # pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
#         # part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
#         # neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
#         # if net == 'RNet':
#         #     landmark_dir = os.path.join('../DATA/imglists_noLM/RNet','landmark_landmark.tfrecord_shuffle')
#         # else:
#         #     landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        
#         # dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
#         # pos_radio = 1.0/6;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=3.0/6
#         # pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
#         # assert pos_batch_size != 0,"Batch Size Error "
#         # part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
#         # assert part_batch_size != 0,"Batch Size Error "
#         # neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
#         # assert neg_batch_size != 0,"Batch Size Error "
#         # landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
#         # assert landmark_batch_size != 0,"Batch Size Error "
#         # batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
#         # #print('batch_size is:', batch_sizes)
#         # image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)  

#     for epoch in range(epochs):
#         for batch in dataset:
#             images, bbox_targets, landmark_targets, labels = batch
#             loss = train_step(images, bbox_targets, landmark_targets, labels, model, optimizer)
#             print(f"Epoch {epoch}, Loss: {loss.numpy()}")      

# if __name__ == '__main__':
#     base_dir = './DATA/imglists/PNet'
#     train(prefix = "./DATA/imglists/PNet",base_dir=base_dir)





