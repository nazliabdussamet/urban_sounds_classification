import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = np.array(np.load("x_train.npy"))
x_test = np.array(np.load("x_test.npy"))
x_val = np.array(np.load("x_val.npy"))
y_train = np.array(np.load("y_train.npy"))
y_test = np.array(np.load("y_test.npy"))
y_val = np.array(np.load("y_val.npy"))


x_train = x_train.reshape(-1, 96, 128, 1)
x_test = x_test.reshape(-1, 96, 128, 1)
x_val = x_val.reshape(-1, 96, 128, 1)


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(110,
                                 kernel_size = (3,3),
                                 strides = (1,1),
                                 padding = "same",
                                 activation = "relu",
                                 input_shape = (96,128,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(110,
                                 kernel_size = (3,3),
                                 strides = (1,1),
                                 padding = "same",
                                 activation = "relu",
                                 input_shape = (96,128,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(220,
                                 kernel_size = (3,3),
                                 strides = (1,1),
                                 padding = "same",
                                 activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(220,
                                 kernel_size = (3,3),
                                 strides = (1,1),
                                 padding = "same",
                                 activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(220,
                                 kernel_size = (3,3),
                                 strides = (1,1),
                                 padding = "same",
                                 activation = "relu"))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(110,
                                activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(220,
                                activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(220,
                                activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10,
                                activation="softmax"))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience = 3)


results = model.fit(x_train, y_train,
                    batch_size=84,
                    epochs=50,
                    callbacks=[callback],
                    validation_data=(x_val,y_val))

plt.plot(results.history["loss"], label="loss")

plt.plot(results.history["val_loss"], label="val_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

plt.plot(results.history["accuracy"], label="accuracy")

plt.plot(results.history["val_accuracy"], label="val_accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

model.evaluate(x_test, y_test)



