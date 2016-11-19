import numpy as np
import scipy
import scipy.fftpack

def extract_features(signal,Wm,sample_rate):
	jump = int(sample_rate*(0.01))
	win_len = int(sample_rate*(0.02))
	L = np.size(signal)
	#print L,win_len,jump
	
	feature = []
	
	for i in range(0,int(L-win_len),int(jump)):
		frame = signal[i:i+win_len]*np.hanning(win_len)
		dft_frame = np.fft.fft(frame,1024)
		pow_dft_frame = (np.abs(dft_frame)**2)/1024
		Ym = np.zeros(34)
		for c in range(0,34):
			Ym[c] = np.sum(np.multiply(Wm[c,:],pow_dft_frame[0:512]))	
		ym = scipy.fftpack.dct(np.log(Ym+1))
		mfcc = ym[0:13]
		mfcc = np.concatenate(([0,0],mfcc[0:13],[0,0]),axis=0)
		delta1 = (mfcc[2:]-mfcc[0:-2])/2
		delta2 = (delta1[2:]-delta1[0:-2])/2

		#feature.append(np.concatenate((mfcc[2:-2],delta1[1:-1],delta2),axis=0))
		feature.append(mfcc[2:-2])


	feature_arr = np.transpose(np.array(feature))

	#avg_mfcc = np.mean(feature_arr,axis=1)
	#cov_mfcc = np.cov(feature_arr)

	#Feat = [avg_mfcc,cov_mfcc]
	
	Feat = feature_arr	
			
	return Feat	