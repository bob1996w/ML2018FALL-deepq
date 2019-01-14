import sys

##########################
### Load Data ###
##########################
dataPath = sys.argv[1]
dataFile = open(dataPath,'r')
dataRawData = dataFile.readlines()
dataFile.close()

dataSize = len(dataRawData)

data = []

for i in range(dataSize):
	if i == 0:
		pass
	else:
		col = dataRawData[i].strip().split(',')
		
		tmp = []
		for j in range(len(col)):
			if j == 0:
				pass
			else:
				tmp.append(float(col[j]))
		data.append(tmp.copy())

minVal = []
maxVal = []

for i in range(14):
	minVal.append(10.0)
	maxVal.append(-10.0)

for i in range(len(data)):
	for j in range(len(data[i])):
		if data[i][j] < minVal[j]:
			minVal[j] = data[i][j]
		if data[i][j] > maxVal[j]:
			maxVal[j] = data[i][j]

for i in range(dataSize):
	if i == 0:
		print(dataRawData[i].strip(), sep="")
	else:
		col = dataRawData[i].strip().split(',')
		
		output = ""
		for j in range(len(col)):
			if j == 0:
				output = output + col[j] + ","
			else:
				output = output + str((float(col[j])-minVal[j-1])/(maxVal[j-1]-minVal[j-1]))
				if j != len(col)-1:
					output = output + ","
				else:
					print(output)


