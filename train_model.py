import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
    model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(), metrics=['accuracy'])

    # Check pointer for storing best model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                   save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    history = model.fit_generator(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        steps_per_epoch=(len(x_train) // train_batchsize) * 2,
        validation_steps=len(x_valid) // valid_batchsize,
        callbacks=[checkpointer, early_stopping],
        verbose=1)

    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')

    # evaluate and print test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])

    model.save("model.h5")