import random
import os
from simulated_channel import loadFilelist

def readDevice(txtfile):

	file1 = open(txtfile, 'r')
	Lines = file1.readlines()
	devices = []
	# Strips the newline character
	for line in Lines:
		devices.append(line.strip().split('/')[-1])

	return devices

def device_degrade(inputPath, outputPath, targetSR, mode='random'):
	# random channel corrupted speech for each utterance
	fileList = loadFilelist(inputPath, '*.wav')

	deviceirlist = './ir-device-file-list.txt'
	spaceirlist = './ir-space-file-list.txt'

	recDevices = readDevice(deviceirlist)
	recSpace = readDevice(spaceirlist)


	for i, inputFile in enumerate(fileList):

		if mode == 'random':
			deviceName = random.choice(recDevices)
			deviceOpt = 'irdevice[filter=' + deviceName + ']'
			outputFile = os.path.join(outputPath, inputFile.split('/')[-1].split('.')[0] + deviceName[:-3] + '.wav')
			cmdFile = './degrade-audio-safe-random.py -D ' + deviceirlist + ' ' + inputFile + ' ' + outputFile
			cmd = cmdFile + ' -r ' + str(targetSR) + ' -c ' + deviceOpt
			os.system(cmd)

		elif mode == 'parallel': 
			devOpts = random.sample(recDevices, 27)
			spcOpts = random.sample(recSpace, 3)

			for deviceName in devOpts:

				deviceOpt = 'irdevice[filter=' + deviceName + ']'

				outputFile = os.path.join(outputPath, inputFile.split('/')[-1].split('.')[0] + deviceName[:-3] + '.wav')
				cmdFile = './degrade-audio-safe-random.py -D ' + deviceirlist + ' ' + inputFile + ' ' + outputFile

				cmd = cmdFile + ' -r ' + str(targetSR) + ' -c ' + deviceOpt

				os.system(cmd)

			for spaceName in spcOpts:

				spaceOpt = 'irspace[filter=' + spaceName + ']'

				outputFile = os.path.join(outputPath, inputFile.split('/')[-1].split('.')[0] + spaceName[:-3] + '.wav')
				cmdFile = './degrade-audio-safe-random.py -D ' + spaceirlist + ' ' + inputFile + ' ' + outputFile

				cmd = cmdFile + ' -r ' + str(targetSR) + ' -c ' + spaceOpt

				os.system(cmd)


if __name__ == "__main__":

	# 
	inputPath = '/home/ge/Documents/Dataset/asvspoof2019/ASVspoof2019_PA_dev/wav'
	outputPath = '/home/ge/Documents/Dataset/asvspoof2019/PA_aug/dev'	
	targetSR = 16000

	device_degrade(inputPath, outputPath, targetSR, mode='parallel')
