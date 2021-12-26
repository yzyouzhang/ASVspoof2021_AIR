import random
import os, fnmatch
from simulated_channel import loadFilelist
import sys

def channel_degrade(inputPath, outputPath, targetSR, mode='random'):
	# random channel corrupted speech for each utterance
	fileList = loadFilelist(inputPath, '*.wav')

	deviceirlist = './ir-device-file-miclist.txt'
	recDevices = readDevice(deviceirlist)

	# codec options
	# landline
	codec_landline = ['g711[law=u]', 'g711[law=a]', 'g726[law=u,br=16k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g726[law=u,40k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]', 'g726[law=a,br=32k]','g726[law=a,br=40k]']

	# voip
	codec_voip = ['silk[br=5k]','silk[br=10k]','silk[br=15k]','silk[br=20k]', \
				'silk[br=5k,loss=5]','silk[br=10k,loss=5]','silk[br=15k,loss=5]','silk[br=20k,loss=5]',\
				'silk[br=5k,loss=10]','silk[br=10k,loss=10]','silk[br=15k,loss=10]','silk[br=20k,loss=10]', \
				'silkwb[br=10k]','silkwb[br=20k]','silkwb[br=30k]', 'silkwb[br=10k,loss=5]',\
				'silkwb[br=20k,loss=5]','silkwb[br=30k,loss=5]','silkwb[br=10k,loss=10]','silkwb[br=20k,loss=10]',\
				'silkwb[br=30k,loss=10]']

	# celluar:
	codec_cell = ['amr[br=4k75]','amr[br=5k15]', 'amr[br=5k9]', 'amr[br=6k7]', \
				'amr[br=7k4]', 'amr[br=7k95]', 'amr[br=10k2]', 'amr[br=12k2]', \
				'amr[br=4k75,nodtx]','amr[br=5k9,nodtx]', 'amr[br=5k9,nodtx]', \
				'amr[br=6k7,nodtx]', 'amr[br=7k4,nodtx]', 'amr[br=7k95,nodtx]', \
				'amr[br=10k2,nodtx]','amrwb[br=6k6]','amrwb[br=12k65]', 'amrwb[br=15k85]', 'amrwb[br=23k05]', \
				'amrwb[br=6k6,nodtx]','amrwb[br=12k65,nodtx]', 'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05,nodtx]']

	# common codec: 2Voip + 1 cell
	codec_common = ['g722[br=64k]', 'g722[br=56k]', 'g722[br=48k]', 'g729a', 'g728', 'gsmfr']


	for inputFile in tqdm(fileList):

		if mode == 'random':
			codecOpts = codec_landline + codec_voip + codec_cell + codec_common
			codecOpt = random.choice(codecOpts)
		elif mode == 'parallel': 
			codecOpts = random.sample(codec_landline, 7) + random.sample(codec_voip, 6) + random.sample(codec_cell, 6) + random.sample(codec_common, 2)

			for codecOpt in codecOpts:

				corOpts = [codecOpt]
				deviceName = random.choice(recDevices)
				devOpt = 'irdevice[filter=' + deviceName + ']'
				corOpts.append(devOpt)
				outputFile = os.path.join(outputPath, inputFile.split('/')[-1].split('.')[0] + '_' + codecOpt + '_' + deviceName + '.wav')

				cmdFile = './degrade-audio-safe-random.py -D ' + deviceirlist + ' ' + inputFile + ' ' + outputFile
				cmd = cmdFile + ' -r ' + str(targetSR) + ' -c ' + ':'.join(corOpts)
				
				os.system(cmd)

if __name__ == "__main__":

	wavPath = '/home/ge/Documents/Dataset/asvspoof2019/ASVspoof2019_LA_train/wav'   
	outputPath = '/home/ge/Documents/Dataset/asvspoof2019/LAPA_aug/train'	
	targetSR = 16000

	# degrade by the channel
	channel_degrade(wavPath, outputPath, targetSR, mode='parallel')

