from keras.models import load_model 
import numpy as np
import os
import wave

encoder = load_model('latest-encoder.h5')
corpusPath = '1s corpus'

songLength = 12000

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
	return data

print('loading data')
data = loadData(corpusPath)

for i, datum in enumerate(data):
	print(str(i) + ': ' + str(encoder.predict(np.array([datum]))[2]))

