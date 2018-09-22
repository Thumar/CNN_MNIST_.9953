import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1)  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images.reshape(mnist.test.images.shape[0], 28, 28, 1)  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

num_class = 10

train_labels = tf.keras.utils.to_categorical(train_labels, num_class)
eval_labels = tf.keras.utils.to_categorical(eval_labels, num_class)

model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, padding='same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.20))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.30))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.40))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.50))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output"))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

mnist_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="kkt-16-adam")

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"conv2d_input": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

#
# mnist_estimator.train(
#     input_fn=train_input_fn,
#     steps=10000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"conv2d_input": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False
)

eval_results = mnist_estimator.evaluate(input_fn=eval_input_fn)
print(eval_results)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"conv2d_input": eval_data[1:45]},
    num_epochs=1,
    shuffle=False
)

predict_result = list(mnist_estimator.predict(input_fn=predict_input_fn))
import matplotlib.pyplot as plt
pos = 1
for img, lbl, predict_lbl in zip(eval_data[1:45], eval_labels[1:45], predict_result):
    output = np.argmax(predict_lbl.get('output'), axis=None)
    lbl = np.argmax(lbl, axis=None)
    plt.subplot(4, 11, pos)
    plt.imshow(img.reshape(28, 28))
    plt.axis('off')
    if output == lbl:
        plt.title(output)
    else:
        plt.title(output + "/" + lbl, color='#ff0000')
    pos += 1

plt.show()
