from keras.layers import Dense, Input, Lambda
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras import backend as K
import numpy as np
import os
import wave

corpusPath = '1s corpus'

batchSize = 4096 
epochs = 1000
songLength = 12000
intermediateDim = 2000
latentDim = 4

def loadData(corpusPath):
	data = []
	for filename in os.listdir(corpusPath):
		if filename.endswith('.wav'):
			w = wave.open(os.path.join(corpusPath, filename))
			song = []
			# get the first <songLength> frames
			frames = w.readframes(songLength)
			[song.append(frame) for frame in frames]
			data.append(song)
			w.close()
	data = np.array(data, dtype=np.float32)
	data = data / 255
	cutoff = int(0.9 * len(data))
	return (data[:cutoff], data[cutoff:])	

def sampling(args):
	zMean, zLogVar = args
	batch = K.shape(zMean)[0]
	dim = K.int_shape(zMean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return zMean + K.exp(0.5 * zLogVar) * epsilon

print('loading data')
train, test = loadData(corpusPath)

print('building model')
# Build encoder
inputs = Input(shape=(songLength, ))
x = Dense(intermediateDim, activation='relu')(inputs)
zMean = Dense(latentDim)(x)
zLogVar = Dense(latentDim)(x)
z = Lambda(sampling)([zMean, zLogVar])

# Instantiate encoder
encoder = Model(inputs, [zMean, zLogVar, z])
encoder.summary() 

# Build decoder
latentInputs = Input(shape=(latentDim, ))
x = Dense(intermediateDim, activation='relu')(latentInputs)
outputs = Dense(songLength, activation='sigmoid')(x)

# Instantiate decoder
decoder = Model(latentInputs, outputs)
decoder.summary()

# Instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)

reconstructionLoss = mse(inputs, outputs)
reconstructionLoss *= songLength
klLoss = 1 + zLogVar - K.square(zMean) - K.exp(zLogVar)
klLoss = K.sum(klLoss, axis=-1)
klLoss *= -0.5
vaeLoss = K.mean(reconstructionLoss + klLoss)
vae.add_loss(vaeLoss)
vae.compile(optimizer='adam')
vae.summary()

print('training model')
vae.fit(train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data=(test, None))

print('saving models')
encoder.save('latest-encoder.h5')
decoder.save('latest-decoder.h5')
vae.save('latest-vae.h5')
