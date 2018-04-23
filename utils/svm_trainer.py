from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np


class SVMTrainer(object):

	def __init__(self, decimal_precision):
		self.precision = decimal_precision
		np.set_printoptions(threshold = np.inf)
		self.svc = SVC()

	def read_data(self, filename):
		with open(filename, 'r') as f:
			lines = f.readlines()
		return np.around(np.array([x.strip().split(' ') for x in lines]).astype(float), decimals = self.precision)

	def scale_data(self, train_data):
		self.scaler = preprocessing.StandardScaler().fit(train_data)
		return self.scaler.transform(train_data)

	def train(self, features, labels):
		non_zero = np.count_nonzero(labels != 0)
		print('[INFO] ', non_zero, ' links present in training data, out of ', len(labels))
		
		self.svc.fit(features, labels)

	def predict(self, file):
		data = self.read_data(file)
		features = data[:, 0:-1]
		labels = data[:, -1].astype(int)
		
		print('[INFO] ', np.count_nonzero(labels != 0), 'links present in training data, out of ', len(labels))

		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			print('[INFO] Round of testing')
			self.train(features[train], labels[train])
			
			preds = self.svc.predict(features[test])
			print('[INFO] Expected: ', labels[test])
			print('[INFO] Result: ', preds)
			
			succ_pred = np.count_nonzero(preds == labels[test])
			print('[INFO] Succesfully predicted ', succ_pred, ' links out of ', len(labels[test]))

	def validate(self, file):
		data = self.read_data(file)
		features = data[:, 0:-1]
		labels = data[:, -1].astype(int)

		k_fold = KFold(n_splits = 3)
		cvr = [self.svc.fit(features[train], labels[train]).score(features[test], labels[test]) 
			for train, test in k_fold.split(features)]

		cvr = cross_val_score(self.svc, features, labels, cv = k_fold, n_jobs = -1)	

		print('[INFO] Cross validation reasults: ', cvr)

def main():
	my_trainer = SVMTrainer(5)
	#my_trainer.validate('data_raw.txt')
	my_trainer.predict('data_raw.txt')
	print('test')

if __name__ == "__main__":
	main()