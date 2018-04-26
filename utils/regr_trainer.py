from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
import numpy as np
import sys

class Regression(object):

	def __init__(self, decimal_precision):
		np.set_printoptions(threshold = np.inf)
		
		self.precision = decimal_precision
		self.regr = LogisticRegression(class_weight = 'balanced')

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
		self.regr.fit(features, labels)

	def predict(self, file):
		data = self.read_data(file)
		features = data[:, 0:-1]
		labels = data[:, -1].astype(int)
		
		print('[INFO] ', np.count_nonzero(labels != 0), 'links present in training data, out of ', len(labels))

		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			print('[INFO] Round of testing')
			self.train(features[train], labels[train])
			
			preds = self.regr.predict(features[test])
			print('[INFO] Expected: ', labels[test])
			print('[INFO] Result: ', preds)
			
			succ_pred = np.count_nonzero(preds == labels[test])
			print('[INFO] Succesfully predicted ', succ_pred, ' links out of ', len(labels[test]))

	def validate(self, file):
		data = self.read_data(file)
		features = data[:, 0:-1]
		labels = data[:, -1].astype(int)

		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.regr.predict(features[test])
			
			print(classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))

def main(argv):
	my_trainer = Regression(17)
	#my_trainer.validate('data_raw.txt')
	#my_trainer.validate('corpus_scores\\10_opt_raw.txt')
	my_trainer.validate('data_raw.txt')
	#my_trainer.predict('data_raw.txt')

if __name__ == "__main__":
	main(sys.argv[1:])