from keras.models import load_model
import numpy as np
import wave

decoder = load_model('latest-decoder.h5')
outputFolder = 'generated/'
n = 5

def generate(l1, l2, l3, l4):
	sample = np.array([[l1, l2, l3, l4]])
	data = decoder.predict(sample)
	name = str(l1) + '_' + str(l2) + '_' + str(l3) + '_' + str(l4) + '.wav'
	w = wave.open(outputFolder + name, 'wb')
	w.setparams((1, 1, 16000, 12000, 'NONE', 'NONE'))
	# data starts as floats (-1, 1)
	data += 1  # (0, 2)
	data /= 2  # (0, 1)
	data *= 255 # (0, 255)
	data = data.astype(np.uint8) # reduce value from 4 bytes to 1 byte
	w.writeframes(data)
	w.close()

l1 = np.linspace(-5, 5, n)
l2 = np.linspace(-5, 5, n)
l3 = np.linspace(-5, 5, n)
l4 = np.linspace(-5, 5, n)

for i, a in enumerate(l1):
	print('Iteration ' + str(i) + ' of first latent variable')
	for b in l2:
		for c in l3:
			for d in l4:
				generate(a, b, c, d)
