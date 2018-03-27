import numpy as np
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

if __name__ == "__main__":
    # Load train data
    x_train = np.load("train_data/x_train.npy") / 255.0
    y_train = np.load("train_data/y_train.npy") / 255.0
    x_test = np.load("train_data/x_test.npy") / 255.0
    y_test = np.load("train_data/y_test.npy") / 255.0
    x_valid = np.load("train_data/x_valid.npy") / 255.0
    y_valid = np.load("train_data/y_valid.npy") / 255.0

    # Data generator
    train_datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    validation_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_batchsize = 32
    valid_batchsize = 10

    train_generator = train_datagen.flow(x_train, y_train, batch_size=train_batchsize)
    validation_generator = validation_datagen.flow(x_valid, y_valid, batch_size=valid_batchsize)

    # Use cpu, gpu out of memory
    model = Sequential()
    model.add(layers.Conv2D(filters=8, kernel_size=2, padding='same', activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(), metrics=['accuracy'])

    # Check pointer for storing best model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                   save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(patience=1, verbose=1, factor=0.8, epsilon=0)
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)'
    # keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
    # https://iwatobipen.wordpress.com/2016/11/23/remotemonitor-in-keras/
    # TODO changed: factor to .7 epsion to 0 patience to 5 for reduce lr: epochs to 100: early stop patience to 10
    history = model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        steps_per_epoch=len(x_train) // train_batchsize,
        validation_steps=len(x_valid) // valid_batchsize,
        callbacks=[checkpointer, early_stopping, reduce_lr],
        verbose=1)

    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')

    # evaluate and print test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])

    model.save("model_batch.h5")
