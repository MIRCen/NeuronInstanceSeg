from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import timeit
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger  #, TensorBoard
from loss import dice_coef_rounded_ch0, dice_coef_rounded_ch1, schedule_steps, softmax_dice_loss
import tensorflow.keras.backend as K
import pandas as pd
from tqdm import tqdm
from transforms import aug_mega_hardcore
from tensorflow.keras import metrics
from abc import abstractmethod
from tensorflow.keras.preprocessing.image import Iterator
import time
from efficientunet import *
import argparse

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser()
parser.add_argument('-data', help='input training dataset folder', type=str)
parser.add_argument('-model', help='output model path', type=str)
args = parser.parse_args()

data_folder = args.data
masks_folder = path.join(data_folder, 'masks_all')
images_folder = path.join(data_folder, 'images_all')
models_folder = args.model

input_shape = (224, 224)

df = pd.read_csv(path.join(data_folder, 'folds.csv'))

all_ids = []
all_images = []
all_masks = []

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

class BaseMaskDatasetIterator(Iterator):
    def __init__(self,
                 image_ids,
                 random_transformers=None,
                 batch_size=8,
                 shuffle=True,
                 seed=None
                 ):
        self.image_ids = image_ids
        self.random_transformers = random_transformers
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(BaseMaskDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    @abstractmethod
    def transform_mask(self, mask, image):
        raise NotImplementedError

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            _idx = self.image_ids[image_index]
            
            img0 = all_images[_idx].copy()
            msk0 = all_masks[_idx].copy()
    
            x0 = random.randint(0, img0.shape[1] - input_shape[1])
            y0 = random.randint(0, img0.shape[0] - input_shape[0])
            img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
            msk = msk0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
            
            data = self.random_transformers[0](image=img[..., ::-1], mask=msk)
                
            img = data['image'][..., ::-1]
            msk = data['mask']
            
            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk

            batch_x.append(img)
            batch_y.append(otp)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")


        batch_x = preprocess_inputs(batch_x)

        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x


    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)    
    
def val_data_generator(val_idx, batch_size, validation_steps):
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            img = all_images[i]
            msk = all_masks[i].copy()            
            msk = msk.astype('float')
            msk[..., 0] = (msk[..., 0] > 127) * 1
            msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 0] == 0) * 1
            msk[..., 2] = (msk[..., 1] == 0) * (msk[..., 0] == 0) * 1
            otp = msk

            for j in range(batch_size):
                inputs.append(img)
                outputs.append(otp)
            if len(inputs) == batch_size:
                step_id += 1
                inputs = np.asarray(inputs)
                outputs = np.asarray(outputs, dtype='float')

                inputs = preprocess_inputs(inputs)

                yield inputs, outputs
                inputs = []
                outputs = []
                if step_id == validation_steps:
                    break
                

def is_grayscale(image):
    return np.allclose(image[..., 0], image[..., 1], atol=0.001) and np.allclose(image[..., 1], image[..., 2], atol=0.001)
                
if __name__ == '__main__':
    t0 = timeit.default_timer()
    # for cross validation
    # fold_nums = [0, 1, 2, 3]
    fold_nums = [1]
    
    if not path.isdir(models_folder):
        mkdir(models_folder)
    
    all_ids = df['img_id'].values
    
    for i in tqdm(range(len(all_ids))):
        img_id = all_ids[i]
        msk = cv2.imread(path.join(masks_folder, '{0}.png'.format(img_id)), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(path.join(images_folder, '{0}.png'.format(img_id)), cv2.IMREAD_COLOR)

        all_images.append(img)
        all_masks.append(msk)
            
    batch_size = 8
    val_batch = 1

    
    for it in range(4):
        if it not in fold_nums:
            continue

        train_idx = df[(df['fold'] != it)].index.values
        train_ids = df[(df['fold'] != it)]['img_id'].values
        train_idx = np.asarray(train_idx)
        
        val_idx = df[(df['fold'] == it)].index.values
        val_idx = np.asarray(val_idx) 
        
        validation_steps = len(val_idx)
        steps_per_epoch = 5 * int(len(train_idx) / batch_size)
        start_epoch = 0
        try:
            log_data = pd.read_csv(path.join(models_folder, 'efficient_b5_weights_{0}.log'.format(it)))
            start_epoch = log_data['epoch'].iloc[-1]
        except FileNotFoundError:
            print("resume failed, log file not found")

        print('Training fold', it)
        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

        data_gen = BaseMaskDatasetIterator(train_idx,
                     random_transformers=[aug_mega_hardcore((-0.25, 0.6)), aug_mega_hardcore((-0.6, 0.25))],
                     batch_size=batch_size,
                     shuffle=True,
                     seed=1
                     )

        
        np.random.seed(it+111)
        random.seed(it+111)
        tf.random.set_seed(it+111)
        
        log_file = path.join(models_folder, 'efficient_b5_weights_{0}.log'.format(it))           
        csv_logger = CSVLogger(log_file, separator=',', append=False)
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))

        model = get_efficient_unet_b5((224, 224, 3), 3, pretrained=True, block_type='transpose', concat_input=True)
        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=3e-4, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])
        model.fit(data_gen,
                                use_multiprocessing=True,
                                epochs=6, steps_per_epoch=steps_per_epoch, verbose=1,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule],
                                max_queue_size=5,
                                workers=6)

        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 10), (1e-4, 40), (5e-5, 55), (2e-5, 65), (1e-5, 70)]))
        for l in model.layers:
            l.trainable = True

        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=5e-6, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])

        model_checkpoint = ModelCheckpoint(path.join(models_folder, 'efficient_b5_weights_{0}.h5'.format(it)), monitor='val_loss', 
                                            save_best_only=True, save_weights_only=True, mode='min')                                            
        model.fit(data_gen,
                                use_multiprocessing=True,
                                epochs=70, steps_per_epoch=steps_per_epoch, verbose=1,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule, model_checkpoint, csv_logger],
                                max_queue_size=5,
                                workers=6)

        del model
        del model_checkpoint
        K.clear_session()
        
        np.random.seed(it+222)
        random.seed(it+222)
        tf.random.set_seed(it+222)
        
        model = get_efficient_unet_b5((224, 224, 3), 3, pretrained=False, block_type='transpose', concat_input=True)
        model.load_weights(path.join(models_folder, 'efficient_b5_weights_{0}.h5'.format(it)))
        lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-6, 72), (3e-5, 80), (2e-5, 90), (1e-5, 100)]))

        model.compile(loss=softmax_dice_loss,
                        optimizer=Adam(lr=1e-5, amsgrad=True),
                        metrics=[dice_coef_rounded_ch0, dice_coef_rounded_ch1, metrics.categorical_crossentropy])

        model_checkpoint2 = ModelCheckpoint(path.join(models_folder, 'efficient_b5_weights_{0}.h5'.format(it)), monitor='val_loss', 
                                            save_best_only=True, save_weights_only=True, mode='min')                                            
        model.fit(data_gen,
                                use_multiprocessing=True,
                                epochs=100, steps_per_epoch=steps_per_epoch, verbose=1,
                                validation_data=val_data_generator(val_idx, val_batch, validation_steps),
                                validation_steps=validation_steps,
                                callbacks=[lrSchedule, model_checkpoint2],
                                max_queue_size=5,
                                workers=6,
                                initial_epoch=92)
        
        del model
        del model_checkpoint2
        K.clear_session()
        
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))