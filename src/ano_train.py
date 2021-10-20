import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.experimental.set_memory_growth(physical_devices[2], True)
tf.config.experimental.set_memory_growth(physical_devices[3], True)
tf.config.experimental.set_memory_growth(physical_devices[4], True)
tf.config.experimental.set_memory_growth(physical_devices[5], True)
tf.config.experimental.set_memory_growth(physical_devices[6], True)
tf.config.experimental.set_memory_growth(physical_devices[7], True)



import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from keras.layers import TimeDistributed, Conv2D, LeakyReLU, BatchNormalization, Dense, ConvLSTM2D, Conv2DTranspose, \
    Input, GlobalAveragePooling2D,SeparableConv2D
import keras
from keras.models import Model
import keras.backend as K
from keras.activations import tanh
from keras import backend
from keras.constraints import Constraint
from keras.optimizers import RMSprop
from PIL import Image
import numpy as np
from imutils import paths
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image


# path to save model
model_dir = "/path/"
# path to save generated image during training
image_dir = "/path/"
# path to trianing videos
video_dir = "Train"


image_types = (".mp4", ".avi")
def list_videos(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath



## list of all training video paths
videoPaths = list(list_videos(video_dir))
all_frames = []

for path in videoPaths:
    cap = cv2.VideoCapture(path)
    w = 128
    h = 128
    fc = 0
    ret = True

    while True:
        ret, frame = cap.read()
        if ret == True:
            resized_frame = cv2.resize(frame, (128, 128))
            all_frames.append(np.array(resized_frame, dtype=np.float64)/256.0)
        else:
            break
    cap.release()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

## get all the training frames
all_frames = np.array(all_frames)


def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The desired distance between two consecutive frames
    frames_list : list
        A list of sorted frames of shape 128 X 128
    sequence_size: int
        The size of the desired LSTM sequence
    Returns
    -------
    list
        A list of clips , 7 frames each
    """
    clips = []
    sz = len(frames_list)
    # print(sz)
    clip = np.zeros(shape=(sequence_size, 128, 128, 3))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, :] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips


## prepare data as needed by the lstm
def dataloader(all_frames):
    clips = []
    for stride in range(1, 3):
        clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=7))
    return clips


training_set = dataloader(all_frames)
training_set = np.array(training_set)
training_set = training_set.reshape(-1, 7, 128, 128, 3)
print("[INFO] data loaded")


## first encoder
input_layer = Input(shape=(7, 128, 128, 3))
x = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
x = BatchNormalization()(x)
sc1 = tanh(x)
x = TimeDistributed(SeparableConv2D(64, (3, 3), strides=2, padding="same", kernel_regularizer='l2'))(sc1)
x = BatchNormalization()(x)
sc2 = tanh(x)
x = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(sc2)
x = BatchNormalization()(x)
x = tanh(x)
x = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(x)
x = BatchNormalization()(x)
sc3 = tanh(x)
x = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(sc3)
x = BatchNormalization()(x)
x = tanh(x)
x = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(x)
x = BatchNormalization()(x)
x = tanh(x)
g_e = Model(inputs=input_layer, outputs=x)


## first decoder
y = g_e(input_layer)
y = TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = keras.layers.Add()([sc2, y])
y = TimeDistributed(Conv2DTranspose(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = keras.layers.Add()([sc1, y])
y = TimeDistributed(Conv2DTranspose(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
y = TimeDistributed(SeparableConv2D(3, (5, 5), activation="tanh", padding="same", kernel_regularizer='l2'))(y)
y = BatchNormalization()(y)
y = tanh(y)
g = Model(inputs=input_layer, outputs=y)




## encoding the generated image
input_layer = Input(shape=(7, 128, 128, 3))
z = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
z = BatchNormalization()(z)
z = tanh(z)
z = TimeDistributed(SeparableConv2D(64, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
z = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(z)
z = BatchNormalization()(z)
z = tanh(z)
encoder = Model(inputs=input_layer, outputs=z)






## feature extractor for descriminator
input_layer = Input(shape=(7, 128, 128, 3))
f = TimeDistributed((SeparableConv2D(128, (5, 5), strides=2, padding="same", kernel_regularizer='l2')))(input_layer)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = TimeDistributed(SeparableConv2D(64, (5, 5), strides=2, padding="same", kernel_regularizer='l2'))(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(64, (3, 3), padding="same", strides=2, return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True,kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
f = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True, kernel_regularizer='l2')(f)
f = BatchNormalization()(f)
f = LeakyReLU()(f)
feature_extractor = Model(inputs=input_layer, outputs=f)



## adversial loss
class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


## Contextual loss
class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

## Encoder loss
class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

## SVD loss
class SVDLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SVDLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        u, s, vt = tf.linalg.svd(ori, full_matrices=False)
        s = tf.linalg.diag(s)
        u1, s1, vt1 = tf.linalg.svd(gan, full_matrices=False)
        s1 = tf.linalg.diag(s1)
        return K.mean(K.abs(u - u1)) + K.mean(K.abs(s - s1)) + K.mean(K.abs(vt - vt1))
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)



gan = g(input_layer)
adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])
svd_loss = SVDLoss(name='svd_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss, svd_loss])

# loss function
def loss(yt, yp):
    return yp


losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
    'svd_loss': loss
}

lossWeights = {'cnt_loss': 20.0, 'adv_loss': 1.0, 'enc_loss': 1.0, 'svd_loss': 1.0}
gan_trainer.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)


## descriminator 
f = feature_extractor(input_layer)
d = TimeDistributed(GlobalAveragePooling2D(name='glb_avg'))(f)
d = TimeDistributed(Dense(1, activation='sigmoid', name='d_out'))(d)
d = Model(input_layer, d)
opt = RMSprop(lr=0.00005)
d.compile(optimizer=opt, loss='binary_crossentropy')


niter = 90000
bz = 3


## generate training samples in batch
def get_data_generator(data, batch_size=3):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen / batch_size))):
            train_x = np.take(data, idxes[i * batch_size: (i + 1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y,y]


train_data_generator = get_data_generator(training_set, bz)

sample_display_image = training_set[:2]


## function to display generated images during training
def plot_save(count, g):
    gan_x = g.predict(sample_display_image)
    test_image = gan_x[0, 1, :, :, :]
    test_image = np.reshape(test_image, (128, 128,3))
    minv=np.min(test_image)
    maxv=np.max(test_image)
    new_image=(test_image-minv)/(maxv-minv)
    filename = image_dir + "Reconstructed_image" + str(count) + ".png"
    if new_image.min()>=0.0 and new_image.max() <=1.0:
        plt.imsave(filename, new_image)
    else:
        print("new image internal  {}".format(new_image))


d_loss_history = []
print(type(d_loss_history))
for i in range(niter):

    ### get batch x, y ###
    x, y = train_data_generator.__next__()
    ### train disciminator ###
    d.trainable = True

    fake_x = g.predict(x)

    d_x = np.concatenate([x, fake_x], axis=0)
    a = np.zeros((len(x), 7, 1))
    b = np.ones((len(fake_x), 7, 1))
    d_y = np.concatenate([a, b])
    d_loss = d.train_on_batch(d_x, d_y)

    ### train generator ###

    d.trainable = False
    g_loss = gan_trainer.train_on_batch(x, y)

    if i % 50 == 0:
        name1 = model_dir + 'g_e' + str(i) + '.h5'
        name2 = model_dir + 'g' + str(i) + '.h5'
        g.save(name2)
        print("[INFO] model saved")
        print(f'niter: {i + 1}, g_loss: {g_loss}, d_loss: {d_loss}')
        plot_save(i, g)
        print("[INFO] image saved")
        d_loss_history = np.array(d_loss_history)
        np.save("d_loss_history", d_loss_history)
        print("loss saved")


## save losss history
d_loss_history = np.array(d_loss_history)
np.save("d_loss_history", d_loss_history)
print("[INFO] training done!")
