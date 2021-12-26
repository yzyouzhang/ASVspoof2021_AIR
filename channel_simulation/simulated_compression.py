import random
from simulated_channel import loadFilelist
import os, fnmatch
from tqdm import tqdm

def compression_degrade(inputPath, outputPath, targetSR, mode='random'):
	# random channel corrupted speech for each utterance
	fileList = loadFilelist(inputPath, '*.wav')

	# compression options
	mp3Parms = ['mp3[8k]','mp3[16k]','mp3[32k]']
	aacParms = ['aac[8k]','aac[16k]','aac[32k]']
	levels = [-26, -29, -32, -35]  # dbFS
	
	compressOpts = mp3Parms + aacParms

	for i, inputFile in enumerate(fileList):

		if mode == 'random':
			compressOpt = random.choice(compressOpts)
		elif mode == 'parallel': 

			for compressOpt in compressOpts:

				level = random.choice(levels)
				codecList=['norm[rms='+ str(level) + ']']

				codecList.append(compressOpt)

				outputFile = os.path.join(outputPath, inputFile.split('/')[-1].split('.')[0] + '_' + compressOpt + '.wav')

				cmdFile = './degrade-audio-safe-random.py ' + inputFile + ' ' + outputFile
				cmd = cmdFile + ' -r ' + str(targetSR) + ' -c ' + ':'.join(codecList)
				os.system(cmd)

if __name__ == "__main__":

	wavPath = '/home/ge/Documents/Dataset/asvspoof2019/ASVspoof2019_LA_train/wav'   
	outputPath = '/home/ge/Documents/Dataset/asvspoof2019/DF_aug/train'	
	targetSR = 16000


	# degrade by the channel
	compression_degrade(wavPath, outputPath, targetSR, mode='parallel')


