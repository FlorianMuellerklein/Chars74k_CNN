import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler

from keras.regularizers import l2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD

from random import randint, uniform

import seaborn as sns
from matplotlib import pyplot
from skimage.io import imshow
from skimage import transform, filters, exposure

PIXELS = 64
imageSize = PIXELS * PIXELS
num_features = imageSize
label_enc = LabelEncoder()

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

def batch_iterator(data, y, batchsize, model):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -10 and 10 degrees
    and to random translations between -5 and 5 pixels in all directions.
    Random zooms between 1 and 1.3.
    Random shearing between -20 and 20 degrees.
    Randomly applies sobel edge detector to 1/4th of the images in each batch.
    Randomly inverts 1/2 of the images in each batch.
    '''

    n_samples = data.shape[0]
    loss = []
    for i in range((n_samples + batchsize -1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.empty(shape = (X_batch.shape[0], 1, PIXELS, PIXELS), dtype = 'float32')

        # random rotations betweein -8 and 8 degrees
        dorotate = randint(-10,10)

        # random translations
        trans_1 = randint(-10,10)
        trans_2 = randint(-10,10)

        # random zooms
        zoom = uniform(1, 1.3)

        # shearing
        shear_deg = uniform(-25, 25)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (trans_1, trans_2))

        tform = tform_center + tform_aug + tform_uncenter

        # images in the batch do the augmentation
        for j in range(X_batch.shape[0]):

            X_batch_aug[j][0] = fast_warp(X_batch[j][0], tform,
                                          output_shape = (PIXELS, PIXELS))

        # use sobel edge detector filter on one quarter of the images
        indices_sobel = np.random.choice(X_batch_aug.shape[0], X_batch_aug.shape[0] / 4, replace = False)
        for k in indices_sobel:
            img = X_batch_aug[k][0]
            X_batch_aug[k][0] = filters.sobel(img)

        # invert half of the images
        indices_invert = np.random.choice(X_batch_aug.shape[0], X_batch_aug.shape[0] / 2, replace = False)
        for l in indices_invert:
            img = X_batch_aug[l][0]
            X_batch_aug[l][0] = np.absolute(img - np.amax(img))

        # fit model on each batch
        loss.append(model.train(X_batch_aug, y_batch))

    return np.mean(loss)

def load_data_cv(train_path):

    print('read data')
    # reading training data
    training = np.load(train_path)

    # split training labels and pre-process them
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    training_targets = np_utils.to_categorical(training_targets)

    # split training inputs and scale data 0 to 1
    training_inputs = training[:,0:num_features].astype('float32')
    training_inputs = training_inputs / np.amax(training_inputs)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(training_inputs, training_targets)

    # reshaping training and testing data so it can be feed to convolutional layers
    x_train = x_train.reshape(x_train.shape[0], 1, PIXELS, PIXELS)
    x_test = x_test.reshape(x_test.shape[0], 1, PIXELS, PIXELS)

    return x_train, x_test, y_train, y_test

def load_data_test(train_path, test_path):

    print('read data')
    # reading training data
    training = np.load(train_path)

    # split training labels and pre-process them
    training_targets = training[:,num_features]
    training_targets = label_enc.fit_transform(training_targets)
    training_targets = training_targets.astype('int32')
    training_targets = np_utils.to_categorical(training_targets)

    # split training inputs and scale data 0 to 1
    training_inputs = training[:,0:num_features].astype('float32')
    training_inputs = training_inputs / np.amax(training_inputs)

    # read testing data
    testing_inputs = np.load(test_path).astype('float32')

    # reshaping training and testing data so it can be feed to convolutional layers
    training_inputs = training_inputs.reshape(training_inputs.shape[0], 1, PIXELS, PIXELS)
    testing_inputs = testing_inputs.reshape(testing_inputs.shape[0], 1, PIXELS, PIXELS)

    return training_inputs, training_targets, testing_inputs

def build_model():
    '''
    VGG style CNN. Using either PReLU or LeakyReLU in the fully connected layers
    '''
    print('creating the model')
    model = Sequential()

    model.add(Convolution2D(128,1,3,3, init='glorot_uniform', activation = 'relu', border_mode='full'))
    model.add(Convolution2D(128,128, 3,3, init='glorot_uniform', activation = 'relu'))
    model.add(MaxPooling2D(poolsize=(2,2)))

    model.add(Convolution2D(256,128, 3,3, init='glorot_uniform', activation = 'relu', border_mode='full'))
    model.add(Convolution2D(256,256, 3,3, init='glorot_uniform', activation = 'relu'))
    model.add(MaxPooling2D(poolsize=(2,2)))

    # convert convolutional filters to flat so they can be feed to fully connected layers
    model.add(Flatten())

    model.add(Dense(65536,2048, init='glorot_uniform', W_regularizer = l2(0.00001)))
    model.add(PReLU(2048))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    model.add(Dense(2048,2048, init='glorot_uniform', W_regularizer = l2(0.00001)))
    model.add(PReLU(2048))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    model.add(Dense(2048,62, init='glorot_uniform'))
    model.add(Activation('softmax'))

    # setting sgd optimizer parameters
    sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def main():

    # switch the commented lines here to alternate between CV testing and making kaggle submission
    x_train, x_test, y_train, y_test = load_data_cv('data/train_32.npy')
    #x_train, y_train, x_test = load_data_test('data/train_32.npy', 'data/test_32.npy')

    model = build_model()

    print("Starting training")
    # batch iterator with 300 epochs
    train_loss = []
    valid_loss = []
    valid_acc = []
    for i in range(300):
        loss = batch_iterator(x_train, y_train, 128, model)
        train_loss.append(loss)
        valid_avg = model.evaluate(x_test, y_test, show_accuracy = True, verbose = 0)
        valid_loss.append(valid_avg[0])
        valid_acc.append(valid_avg[1])
        print 'epoch:', i, 'train loss:', np.round(loss, decimals = 4), 'valid loss:', np.round(valid_avg[0], decimals = 4), 'valid acc:', np.round(valid_avg[1], decimals = 4)

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    valid_acc = np.array(valid_acc)
    sns.set_style("whitegrid")
    pyplot.plot(train_loss, linewidth = 3, label = 'train loss')
    pyplot.plot(valid_loss, linewidth = 3, label = 'valid loss')
    pyplot.legend(loc = 2)
    pyplot.ylim([0,4.5])
    pyplot.twinx()
    pyplot.plot(valid_acc, linewidth = 3, label = 'valid accuracy', color = 'r')
    pyplot.grid()
    pyplot.ylim([0,1])
    pyplot.legend(loc = 1)
    pyplot.show()


    #print("Generating predections")
    #preds = model.predict_classes(x_test, verbose=0)
    #preds = label_enc.inverse_transform(preds).astype(str)

    #submission = pd.read_csv('sampleSubmission.csv', dtype = str)
    #submission['Class'] = preds
    #submission.to_csv('keras_cnn.csv', index = False)

if __name__ == '__main__':
    main()
