import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import sunau
import numpy as np
from os import walk
import os
from ML_functions import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

sample_rate = 22050

# MEL Frequency banks for MFCCs

m1 = 1125*np.log(1+300/700)
m2 = 1125*np.log(1+(sample_rate/2)/700)
			
m = np.linspace(m1,m2,num=36)
h = 700*(np.exp(m/1125)-1)
f = np.floor((1024+1)*h/sample_rate)

Wm = np.zeros([34,512])

for a in range(0,34):
	for b in range(0,512):
		if (b<f[a]):
			Wm[a,b]=0
		elif (b>=f[a] and b<=f[a+1]):
			Wm[a,b] = (b-f[a])/(f[a+1]-f[a])
		elif (b>=f[a+1] and b<f[a+2]):
			Wm[a,b] = (f[a+2]-b)/(f[a+2]-f[a+1])
		else:
			Wm[a,b] = 0


m = []

for (dirpath, dirname, filenames) in walk('genres_new'):
    direc = dirname
    break

q=[] 

k = 0
mydata = []
for directory in direc:
	m = []
	for (dirpath, dirname, filenames) in walk(os.path.join('genres_new',directory)):
		m.extend(filenames)
    	#m.pop(0)
	j=0	
	for file in m:
		file_path = os.path.join('genres_new',directory, file)

		# Reading a single audio file in .au format
		sound = sunau.Au_read(file_path)                                         
		
		# Converting the read file into numpy array 
		p = np.fromstring(sound.readframes(sound.getnframes()),dtype = np.int16)

		F = extract_features(p,Wm,sample_rate)
		F.append(k)
		#F = np.load(file_path)
		#F = F[:,0:3000]
		#print np.shape(F)
		mydata.append(F)
		

		save_path = os.path.join('F_new',str(k), str(j))
		np.save(save_path,F)

		j = j+1

	k = k+1

my_data = np.array(mydata)

# # my_data -> array , mydata -> list

np.save('my_data.npy',my_data)



my_data = np.load('my_data.npy')
my_data = np.array(my_data)
# print np.shape(my_data)


final_data = np.concatenate((my_data[0],my_data[1]),axis=1)
for k in range(2,400):
	final_data = np.concatenate((final_data,my_data[k]),axis=1)

# print np.shape(final_data)

np.save('final_data.npy',final_data)

final_data = np.load('final_data.npy')

kmeans = KMeans(n_clusters=120, random_state=0).fit(np.transpose(final_data))
codebook = kmeans.cluster_centers_

np.save('codebook.npy',codebook)

# np.save('kmeans.npy',kmeans)

codebook = np.load('codebook.npy')

#kmeans = np.load('kmeans.npy')

kmeans = KMeans()
kmeans.cluster_centers_ = codebook

for (dirpath, dirname, filenames) in walk('F_new'):
    direc = dirname
    break


h = 0
k = 0
class_data = []
for directory in direc:
	m = []
	for (dirpath, dirname, filenames) in walk(os.path.join('F_new',directory)):
		m.extend(filenames)
    	#m.pop(0)
	j=0
	for file in m:
		file_path = os.path.join('F_new',directory,file)

		feat = np.load(file_path)

		pred = kmeans.predict(np.transpose(feat))

		fnew = np.zeros([121,1])

		hist,bin_size = np.histogram(pred,bins = 120)
		fnew[120,0] = k
		fnew[0:120,0] = hist

		save_path = os.path.join('Hist',str(k), str(j))
		np.save(save_path,fnew)
		class_data.append(fnew)
		
		h = h+1
		j = j+1
	k = k+1	

	
np.save('class_data.npy',class_data)


class_data = np.load('class_data.npy')

c_dat = np.array(class_data)

N_feat = np.zeros([400,61])
for j in range(0,400):
	b = c_dat[j][:]
	N_feat[j,:] = b[:,0]


# print np.shape(X),np.shape(Y)

train = np.concatenate([N_feat[0:70,:],N_feat[100:170,:],N_feat[200:270,:],N_feat[300:370,:]])
test = np.concatenate([N_feat[70:100,:],N_feat[170:200,:],N_feat[270:300,:],N_feat[370:400,:]])

# train = np.concatenate([N_feat[0:70,:],N_feat[200:270,:]])
# test = np.concatenate([N_feat[70:100,:],N_feat[270:300,:]])

# train = np.concatenate([N_feat[200:270,:],N_feat[300:370,:]])
# test = np.concatenate([N_feat[270:300,:],N_feat[370:400,:]])

train = np.random.permutation(train)
test = np.random.permutation(test)

ytrain = np.uint16(train[:,60])
xtrain = train[:,0:60]

ytest = test[:,60]
xtest = test[:,0:60]

# scaler = StandardScaler()
# scaler.fit(xtrain)
# xtrain = scaler.transform(xtrain)
# xtest = scaler.transform(xtest)

# xtrain = xtrain/3000.0

n1 = 280
n2 = 120

ytrain1 = np.zeros(n1)
ytrain1[ytrain==0]=1

# print ytrain1

ytrain2 = np.zeros(n1)
ytrain2[ytrain==1]=1

ytrain3 = np.zeros(n1)
ytrain3[ytrain==2]=1

ytrain4 = np.zeros(n1)
ytrain4[ytrain==3]=1

ytest1 = np.zeros(n2)
ytest1[ytest==0]=1

ytest2 = np.zeros(n2)
ytest2[ytest==1]=1

ytest3 = np.zeros(n2)
ytest3[ytest==2]=1

ytest4 = np.zeros(n2)
ytest4[ytest==3]=1



clf = SVC(kernel = 'linear',probability=True,random_state=0)
clf.fit(xtrain,ytrain1)
#print clf.classes_
#print clf.score(xtest,ytest1)
#A0 = clf.predict_proba(xtest)
A0 = clf.predict_proba(xtest)
#print A0[119,:]


clf.fit(xtrain,ytrain2)
#print clf.classes_
#print clf.score(xtest,ytest2)
#A1 = clf.predict_proba(xtest)
A1 = clf.predict_proba(xtest)
#print A1[119,:]

clf.fit(xtrain,ytrain3)
#print clf.classes_
#print clf.score(xtest,ytest3)
#A2 = clf.predict_proba(xtest)
A2 = clf.predict_proba(xtest)
#print A2[119,:]


clf.fit(xtrain,ytrain4)
#print clf.classes_
#print clf.score(xtest,ytest4)
#A3 = clf.predict_proba(xtest)
A3 = clf.predict_proba(xtest)
#print A3[119,:]

temp = np.zeros(4)
pred = np.zeros(120)
for i in range(0,120):
	temp[0] = A0[i,1]
	temp[1] = A1[i,1]
	temp[2] = A2[i,1]
	temp[3] = A3[i,1]
	pred[i] = np.argmax(temp)

cor = np.sum(pred==ytest)

print 	cor/120.0

Conf_matrix = np.zeros([4,4])

for i in range(0,n2):
	Conf_matrix[ytest[i],pred[i]]+=1

print Conf_matrix	


## Random Forests

# clf = RandomForestClassifier(n_estimators = 1000,random_state = 1)
# clf.fit(xtrain,ytrain)
# print clf.score(xtest,ytest)


## K Neighbors

# neigh = KNeighborsClassifier(weights='distance',n_neighbors = 5)
# neigh.fit(xtrain,ytrain)
# print neigh.score(xtest,ytest)
# pred = neigh.predict(xtest)


## Neural Network

# n = 50

# mlp = MLPClassifier(hidden_layer_sizes=(n,n,n,n), activation = 'relu', max_iter=100, alpha=1e-5,
#                     solver='lbfgs', verbose=False, tol=1e-4, random_state=1,
#                     learning_rate_init=0.001)


# mlp.fit(xtrain,ytrain)
# print mlp.score(xtest,ytest)




















		
