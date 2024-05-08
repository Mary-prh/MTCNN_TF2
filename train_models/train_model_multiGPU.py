"""
Distributed training across multiple devices (GPUs)
Model: MTCNN
"""
import os
import json
import math
import tensorflow as tf
from MTCNN_config import *
from read_tfrecod_tf2 import read_single_tfrecord
from mtcnn import *
from losses import *
# from keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
# Set log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Train:
    def __init__(self, net, strategy=None):
        self.strategy = strategy if strategy is not None else tf.distribute.get_strategy()
        with self.strategy.scope():
            self.net = net
            self.model      = self.__init_model()
            batch_size = config.BATCH_SIZE
            if tf.distribute.has_strategy():
                self.batch_size = batch_size * self.strategy.num_replicas_in_sync
                print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
            self.base_dir   = config.BASE_DIR
            self.lr_base    = config.LR_BASE
            # self.reduce_lr_callback = self.__my_callback()
            self.optimizer  = self.__init_optimizer()
            self.dataset    = self.__load_dataset(mode='train')
            self.num_batches= self.__define_batches()
            self.classification_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            self.bbox_accuracy_metric = tf.keras.metrics.MeanSquaredError()
            self.landmark_accuracy_metric = tf.keras.metrics.MeanSquaredError()

            self.checkpoint_path = "checkpoints/"
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            self.checkpoint_prefix  = os.path.join(self.checkpoint_path, f"ckpt_{self.net}_")
            self.checkpoint         = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            self.best_checkpoint_path = os.path.join(self.checkpoint_path, f"best_{self.net}")
            self.best_checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            self.best_loss = float("inf")
            self.history = {
                "epochs": [],
                "loss": [],
                "classification_accuracy": [],
                "bbox_accuracy": [],
                "landmark_accuracy": [],
                "learning_rate": []
            }
            self.history_file = os.path.join(self.checkpoint_path, "training_history.json")
            
            self.__train_step = tf.function(self.__train_step)

    def __load_dataset(self,mode):
        dataset_dir = os.path.join(self.base_dir,f'{self.net}/{mode}_{self.net}_landmark.tfrecord_shuffle')
        dataset     = read_single_tfrecord(dataset_dir, self.batch_size, self.net)
        if tf.distribute.has_strategy():
            data_options = tf.data.Options()
            data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            dataset = dataset.with_options(data_options)
        print(f'load dataset from: {self.net}/train_{self.net}_landmark.tfrecord_shuffle')
        return dataset
    
    def __define_batches(self,):
        # Calculate expected batches and compare with actual count
        landmark_dir = os.path.join(self.base_dir,f'{self.net}/train_{self.net}_landmark.txt')
        f = open(landmark_dir, 'r')
        num_samples = len(f.readlines())
        num_batches = math.ceil(num_samples / self.batch_size)
        return num_batches
    
    def __init_model(self,):
        if self.net == "PNet":
            image_size      = get_model_config(net_type= self.net)[0]
            model = create_pnet(image_size,image_size)
            print(f'{self.net} model created')
        else:
            return print('work later!')
        return model
    
    def __init_optimizer(self,):
        optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = self.lr_base, momentum=0.9)
        return optimizer
    
    def __reduce_lr_on_plateau(self, epoch_loss, factor=0.1, patience=3, min_lr=1e-6):
        if self.lr_base > min_lr and epoch_loss >= self.best_loss:
            patience -= 1
            if patience == 0:
                self.lr_base = max(self.lr_base * factor, min_lr)
                self.optimizer.learning_rate.assign(self.lr_base)
                patience = 3
                print(f"Reducing learning rate to {self.lr_base}")
        elif epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
    
    def __convert_sparse_to_dense(self, gradients):
        """Convert sparse gradients to dense for efficient computation."""
        return [
            tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g
            for g in gradients
        ]
    @tf.autograph.experimental.do_not_convert
    def __train_step(self, images, labels, rois, landmarks, optimizer):
        with tf.GradientTape() as tape:
            classifier_output, bbox_output, landmark_output = self.model(images, training=True)

            loss_weights = get_model_config(net_type=self.net)[1]

            # Apply OHEM for classification loss
            loss_classifier = cls_ohem(classifier_output, labels) * loss_weights['classifier']

            # Apply OHEM for bounding box regression loss
            loss_bbox = bbox_ohem(bbox_output, rois, labels) * loss_weights['bbox_regress']

            # Apply OHEM for landmark loss
            loss_landmark = landmark_ohem(landmark_output, landmarks, labels) * loss_weights['landmark_pred']

            # Update metrics
            self.classification_accuracy_metric.update_state(labels, classifier_output)
            self.bbox_accuracy_metric.update_state(rois, bbox_output)
            self.landmark_accuracy_metric.update_state(landmarks, landmark_output)

            total_loss = (loss_classifier + loss_bbox + loss_landmark)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = self.__convert_sparse_to_dense(gradients)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
        return total_loss, loss_classifier, loss_bbox, loss_landmark

    def __distributed_train_step(self, images, labels, rois, landmarks):
            per_replica_losses = self.strategy.run(self.__train_step, args=(images, labels, rois, landmarks, self.optimizer))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def __print_progress(self, iteration, total, loss, prefix='', suffix='', decimals=1, length=100, fill='>'):
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
        print(f'\r{prefix}/{total} |{bar}| {percent}% Loss: {loss:.4f} Classification Accuracy: {self.classification_accuracy_metric.result():.4f} {suffix}', end='\r')
        if iteration == total:
            print()

    def __save_history_to_json(self):
        """Save training history to a JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
            print('Saving training history...')

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            epoch_loss = 0
            num_processed_batches = 0
            epoch_classification_accuracy = 0
            epoch_bbox_accuracy = 0
            epoch_landmark_accuracy = 0
            for i, (images, labels, rois, landmarks) in enumerate(self.dataset):
                # loss_values, _, _, _ = self.__train_step(images=images, labels=labels, rois=rois, landmarks=landmarks, 
                #                         optimizer= self.optimizer)  
                loss_values, _, _, _ = self.__distributed_train_step(images, labels, rois, landmarks)
                
                current_loss = loss_values
                epoch_loss += current_loss
                epoch_classification_accuracy += self.classification_accuracy_metric.result()
                epoch_bbox_accuracy += self.bbox_accuracy_metric.result()
                epoch_landmark_accuracy += self.landmark_accuracy_metric.result()

                num_processed_batches += 1

                self.__print_progress(i + 1, self.num_batches, current_loss, prefix=i, suffix='', length=20)
                
                if num_processed_batches >= self.num_batches:
                    break
            
            average_epoch_loss = epoch_loss / num_processed_batches
            average_epoch_classification_accuracy = epoch_classification_accuracy / num_processed_batches
            average_epoch_bbox_accuracy = epoch_bbox_accuracy / num_processed_batches
            average_epoch_landmark_accuracy = epoch_landmark_accuracy / num_processed_batches

            self.__reduce_lr_on_plateau(average_epoch_loss, factor=0.1, patience=3)

            # Save model weights in h5 format
            h5_checkpoint_prefix = self.checkpoint_prefix + f"{epoch + 1}.h5"
            self.model.save_weights(h5_checkpoint_prefix)
            # Save checkpoint with optimizer
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)


            print(f'/nEpoch {epoch+1}/naverage_epoch_loss: {average_epoch_loss:.4f}/n\
                  average_epoch_classification_accuracy: {average_epoch_classification_accuracy}/n\
                  average_epoch_bbox_accuracy: {average_epoch_bbox_accuracy}/n\
                   average_epoch_landmark_accuracy: {average_epoch_landmark_accuracy} ')
            
            # self.checkpoint.save(file_prefix=self.checkpoint_prefix + f"epoch_{epoch+1}")
            # Check if this is the best model
            if average_epoch_loss < self.best_loss:
                self.best_loss = average_epoch_loss
                self.model.save_weights(self.best_checkpoint_path + ".h5")
                self.best_checkpoint.save(file_prefix=self.best_checkpoint_path)
                print(f"\nNew best model saved with loss: {self.best_loss:.4f}")
            
            # Update history and save to JSON
            self.history["epochs"].append(epoch + 1)
            self.history["loss"].append(average_epoch_loss.numpy().tolist())
            self.history["classification_accuracy"].append(average_epoch_classification_accuracy.numpy().tolist())
            self.history["bbox_accuracy"].append(average_epoch_bbox_accuracy.numpy().tolist())
            self.history["landmark_accuracy"].append(average_epoch_landmark_accuracy.numpy().tolist())
            self.history["learning_rate"].append(self.lr_base)

            self.__save_history_to_json()

        print("Training complete!")


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        trainer = Train(net='PNet', strategy=strategy)
        trainer.train(num_epochs=10)  


    
        

