import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
encoded = Dense(encoding_dim, activation='relu')(encoded)
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

try:
    history=autoencoder.fit(x_train, x_train,
                    epochs=5,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
except Exception as e:
    print("Caught exception:", e)


# Predict on test data
reconstructed_imgs = autoencoder.predict(x_test)

# Choose an index to visualize
idx = 1  # you can change this to try other samples

# Original image (reshape from 784 to 28x28)
original = x_test[idx].reshape(28, 28)
# Reconstructed image
reconstructed = reconstructed_imgs[idx].reshape(28, 28)

# Plot original and reconstructed side by side
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


# Save model
autoencoder.save("autoencoder_model.h5")

# Save training history using pickle
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
