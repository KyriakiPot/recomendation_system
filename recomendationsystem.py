import random
import numpy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
import math
from sklearn.metrics.pairwise import cosine_similarity
import configparser

# Create configuration file

configParser = configparser.RawConfigParser()   
configFilePath = r'c.txt'
configParser.read(configFilePath)
N = int(configParser.get('var', 'N'))
M = int(configParser.get('var', 'M'))
T1 = int(configParser.get('var', 'T'))
T = int((N*M*T1)/100)
T_rest = (N*M)-T
NNeighbors = int(configParser.get('var', 'K'))
R = int(configParser.get('var', 'R'))

f = open("output.txt","w+")

#########################

# Create array

arr = [[0 for i in range(M)] for j in range(N)] 

for i in range(N):
    for j in range(M):
        arr[i][j] = round(random.uniform(1,5),3)
 
x = numpy.array(arr) 

#########################

#Replace 1-Τ elements to 0 (missing values). 

new_array = numpy.copy(x)

k=0
index = []
flag=False

for x in range(N):
	for y in range(M):
		if(random.randint(0,1)):
			flag = True
			if k>T_rest-1 :
				break
			new_array[x][y]= 0
			i=x,y
			k=k+1
		if flag :
			index.append(i)	
			flag=False

			
print("array is ")
print(new_array)
print(" ")
f.write("array \n")
f.write(str(new_array))

#########################

# Calculate pearson correlation

y = numpy.corrcoef(new_array)

pearson = numpy.around(y,2)

#########################

# Search nearest neighbors

nbrs = NearestNeighbors(n_neighbors = NNeighbors,metric = correlation).fit(new_array)
distances, indices = nbrs.kneighbors(new_array)

print("nearest neighbors with pearson correlation are : ") 
print(indices)
print(" ")

#########################

# Search nearest neighbors' pearson correlation 

NNpearson = []
NNpearson = numpy.arange(N* NNeighbors).reshape(N, NNeighbors)
NNpearson = NNpearson.astype(numpy.float)

for i in range(N):
	for j in range(NNeighbors):
		y = indices[i][j]
		NNpearson[i][j] = pearson[i][y]

#########################

# Calculate weighted avarage 
		
x = []
avarage1=[]
avarage1 = numpy.arange(N*M).reshape(N, M)
avarage1 = avarage1.astype(numpy.float)	
for i in range(N):
	for j in range(M):
		sum = 0.0
		dsum = 0.0
		for w in range(NNeighbors):
			x=indices[i][w]
			if (new_array[x][j] != 0):
				sum = sum + (NNpearson[x][w]*new_array[x][j])
				dsum = dsum+NNpearson[x][w]		
		if(dsum != 0):	
			avarage1[i][j] = sum/dsum
			
print("predicted array is ")	
print(avarage1)
print(" ")

#########################

# Search for top rated elements 

avarage1S = -numpy.sort(-avarage1)

sortedAv1 = []
sortedAv1 = numpy.arange(N * R).reshape(N, R)
sortedAv1 = sortedAv1.astype(numpy.float)

for i in range(N):
	for j in range(R):
		sortedAv1[i][j] = avarage1S[i][j]
		
top = []
top = numpy.arange(N * R).reshape(N, R)
		
for i in range(N):
	for j in range(M):
		for w in range(R):
			if(avarage1[i][j] == sortedAv1[i][w]):
				top[i][w] = j

for i in range(N):
	print("User's ",i,"top rated elements are: " ,top[i])
	
#########################
	
print(" ")

# Calculate cosine_similarity

similarity = cosine_similarity(new_array)

#########################

# Search for nearest neighbors

nbrs2 = NearestNeighbors(n_neighbors=NNeighbors,metric = 'cosine').fit(new_array)
distances2, indices2 = nbrs2.kneighbors(new_array)

print("Nearest Neighbors with cosine_similarity are")
print(indices2)
print(" ")

#########################

# Search nearest neighbors' cosine_similarity 

y2 = []
NNsimilarity = []
NNsimilarity = numpy.arange(N*NNeighbors).reshape(N, NNeighbors)
NNsimilarity = NNsimilarity.astype(numpy.float)
for i in range(N):
	for j in range(NNeighbors):
		y2 = indices2[i][j]
		NNsimilarity[i][j] = similarity[i][y2]

#########################

# Calculate weighted avarage

avarage2=[]
avarage2 = numpy.arange(N*M).reshape(N, M)
avarage2 = avarage2.astype(numpy.float)	
for i in range(N):
	for j in range(M):
		sum2 = 0.0
		dsum2 = 0.0
		for w in range(NNeighbors):
			x2=indices2[i][w]
			if (new_array[x2][j] != 0):
				sum2 = sum2 + (NNsimilarity[x2][w] * new_array[x2][j])
				dsum2 = dsum2 + NNsimilarity[x2][w]
			if(dsum2 != 0):
				avarage2[i][j] = sum2/dsum2
			
print("predicted array is ")	
print(avarage2)
print(" ")

#########################

# Search for top rated elements 

avarage2S = -numpy.sort(-avarage2)

sortedAv2 = []
sortedAv2 = numpy.arange(N * R).reshape(N, R)
sortedAv2 = sortedAv2.astype(numpy.float)

for i in range(N):
	for j in range(R):
		sortedAv2[i][j] = avarage2S[i][j]
		
top2 = []
top2 = numpy.arange(N * R).reshape(N, R)
		
for i in range(N):
	for j in range(M):
		for w in range(R):
			if(avarage2[i][j] == sortedAv2[i][w]):
				top2[i][w] = j

for i in range(N):
	print("User's ",i,"top rated elements are: " ,top2[i])
	
print(" ")

#########################

# Search for user's really top rated elements 

really_top = []
really_top = numpy.arange(N * M).reshape(N, M)
rel = 0
relevant = []
relevant = numpy.arange(N).reshape(N, 1)
	
for i in range(N):
	rel = 0
	for j in range(M):
		if(new_array[i][j] >= 3.5):
			really_top[i][j] = j
			rel = rel + 1
		else :
			really_top[i][j] = -1
	
	relevant[i] = rel

for i in range(N):
	for j in range(M):
		if(really_top[i][j] != -1):
			print("User's ",i,"really top rated elements are: " ,really_top[i][j])

#########################	

# Calculate true-positives, false-positives, false-negatives for system B

TPB = []
TPB = numpy.arange(N).reshape(N,1)	
FPB = []
FPB = numpy.arange(N).reshape(N,1)
FNB = []
FNB = numpy.arange(N).reshape(N,1)


for i in range(N):
	counter1=0
	counter2=0
	counter3=0
	for j in range(M):
	
		for w in range(R):
			if(top[i][w] == really_top[i][j]):
				counter1 = counter1 + 1
					
		counter2 = R - counter1
		counter3 = relevant[i] - counter1
	TPB[i] = counter1
	FPB[i] = counter2
	FNB[i] = counter3

#########################	

# Calculate precision and recall for system B

PB = []
PB = numpy.arange(N).reshape(N,1)
PB = PB.astype(numpy.float)
RB = []
RB = numpy.arange(N).reshape(N,1)
RB = RB.astype(numpy.float)

for i in range(N):
	if(TPB[i] != 0):
		PB[i] = TPB[i]/(TPB[i]+FPB[i])
	else :
		PB[i] = 0

for i in range(N):
	if(TPB[i] != 0):
		RB[i] = TPB[i]/(TPB[i]+FNB[i])
	else :
		RB[i] = 0
		
#########################	

# Calculate macro evaluation of F_measure for system B

F_measureB = []
F_measureB = numpy.arange(N).reshape(N,1)
F_measureB = F_measureB.astype(numpy.float)

for i in range(N):
	if(PB[i] != 0 and RB[i] != 0):
		F_measureB[i] = (2*PB[i]*RB[i])/(PB[i]+RB[i])
	else :
		F_measureB[i] = 0
		
#########################		

# Calculate avarage F-measure for system B

F_B = 0
for i in range(N):
	F_B = F_B + F_measureB[i]


F_B = F_B/N	

print(" ")
print("F- measure for system B")		
print(F_B)
f.write("\nF-measure for system B:  ")
f.write(str(F_B))	

#########################

# Calculate precision for system B

precisionB = []
precisionB = numpy.arange(N).reshape(N, 1)	
precisionB = precisionB.astype(numpy.float)

for i in range(N):
	if(relevant[i] != 0):
		counter = 0
		sum=0
		for w in range(R):
			for j in range(M):
				if(top[i][w] == really_top[i][j]):
					counter = counter + 1
					sum = sum+(counter/(w+1))
						
		precisionB[i] = sum/R		
	else :
		precisionB[i]=0
		
#########################	

# Calculate MAP	for system B
sum=0

for i in range(N):
	sum = sum + precisionB[i]
	
sum = sum/N
print(" ")
print("MAP for system B: ",sum)
f.write("\nMAP for system B:  ")
f.write(str(sum))	

#########################	

# Calculate true-positives, false-positives, false-negatives for system C
	
TP = []
TP = numpy.arange(N).reshape(N,1)	
FP = []
FP = numpy.arange(N).reshape(N,1)
FN = []
FN = numpy.arange(N).reshape(N,1)


for i in range(N):
	counter1=0
	counter2=0
	counter3=0
	for j in range(M):
	
		for w in range(R):
			if(top2[i][w] == really_top[i][j]):
				counter1 = counter1 + 1
					
		counter2 = R - counter1
		counter3 = relevant[i] - counter1
	TP[i] = counter1
	FP[i] = counter2
	FN[i] = counter3

#########################	

# Calculate precision and recall for system C	

PC = []
PC = numpy.arange(N).reshape(N,1)
PC = PC.astype(numpy.float)
RC = []
RC = numpy.arange(N).reshape(N,1)
RC = RC.astype(numpy.float)

for i in range(N):
	if(TP[i] != 0):
		PC[i] = TP[i]/(TP[i]+FP[i])
	else :
		PC[i] = 0

for i in range(N):
	if(TP[i] != 0):
		RC[i] = TP[i]/(TP[i]+FN[i])
	else :
		RC[i] = 0

#########################	

# Calculate macro evaluation of F_measure for system C		

F_measureC = []
F_measureC = numpy.arange(N).reshape(N,1)
F_measureC = F_measureC.astype(numpy.float)

for i in range(N):
	if(PC[i] != 0 and RC[i] != 0):
		F_measureC[i] = (2*PC[i]*RC[i])/(PC[i]+RC[i])
	else :
		F_measureC[i] = 0

#########################		

# Calculate avarage F-measure for system C

F_C = 0
for i in range(N):
	F_C = F_C + F_measureC[i]

F_C = F_C/N
print(" ")
print("F- measure for system C")		
print(F_C)
f.write("\nF-measure for system C:  ")
f.write(str(F_C))

#########################

# Calculate precision for system C

precisionC = []
precisionC = numpy.arange(N).reshape(N, 1)	
precisionC = precisionC.astype(numpy.float)
			

	
for i in range(N):
	if(relevant[i] != 0):
		counter = 0
		sum=0
		for w in range(R):
			for j in range(M):
				if(top2[i][w] == really_top[i][j]):
					counter = counter + 1
					sum = sum+(counter/(w+1))
		precisionC[i] = sum/R		
				
	else :
		precisionC[i]=0				

#########################	

# Calculate MAP	for system C

sum=0
for i in range(N):
	sum = sum + precisionC[i]

sum = sum/N
print(" ")
print("MAP for system C: ",sum)
f.write("\nMAP for system C:  ")
f.write(str(sum))

#########################


					
			