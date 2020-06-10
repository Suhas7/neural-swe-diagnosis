import timesynth as ts
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

Features=5
tt_split=.8

class Application:
	def generateData(self):
		time_sampler = ts.TimeSampler(stop_time=50)
		irregular_time_samples = time_sampler.sample_irregular_time(num_points=1000, keep_percentage=50)
		white_noise = ts.noise.GaussianNoise(std=0.3)
		gp = ts.signals.GaussianProcess(kernel='Matern', nu=3./2)
		gp_series = ts.TimeSeries(signal_generator=gp)
		z=[]
		for _ in range(Features):
			z.append(gp_series.sample(irregular_time_samples)[0])
		self.features=np.asarray(z).T
		a = self.features[:,0]*rd.gauss(1,.2)
		for i in range(1,5):
			a = np.add(a,self.features[:,i]*rd.gauss(1,.2))
		self.target=np.asarray([a]).T
		self.data=np.concatenate((self.features.T,self.target.T)).T
	def __init__(self,ID):
		self.ID=ID
		self.parents = []
		self.children = []
		self.generateData()
		self.model=XGBRegressor()
	def addParent(self,other):
		self.parents.append(other)
		self.target=np.add(self.target,other.target*abs(rd.gauss(0,.2)))
	def fitAndTest(self,includeParents=False):
		if not includeParents:
			self.model.fit(self.features[:int(500*tt_split),:],self.target[:int(500*tt_split),:])
			yPred = self.model.predict(self.features[int(500*tt_split):,:])
		else:
			X=self.features
			for par in self.parents:
				X=np.concatenate(X.T,par.target.T).T
			self.model.fit(X[:int(500*tt_split),:],self.target[:int(500*tt_split),:])
			yPred = self.model.predict(X[int(500*tt_split):,:])
		print("ID: {} Score: {}".format(self.ID,mean_squared_error(self.target[int(500*tt_split):,:],yPred)))
