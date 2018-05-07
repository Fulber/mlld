from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from data_reader import DataReader as dr
import numpy as np

class SVMTrainer(object):

	def __init__(self, decimal_precision):
		np.set_printoptions(threshold = np.inf)
		
		self.precision = decimal_precision
		self.svc = SVC(kernel = 'rbf', class_weight = 'balanced')

	def scale_data(self, train_data):
		self.scaler = preprocessing.StandardScaler().fit(train_data)
		return self.scaler.transform(train_data)

	def train(self, features, labels):
		non_zero = np.count_nonzero(labels != 0)
		print('[INFO] ', non_zero, ' links present in training data, out of ', len(labels))
		self.svc.fit(features, labels)

	def predict(self, file):
		data = dr(file).read_data(precision = self.precision)
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
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.svc.predict(features[test])
		
			print(classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))

def main():
	my_trainer = SVMTrainer(-1)
	#my_trainer.validate('data_raw.txt')
	my_trainer.validate('corpus_scores\\v2_5_raw.txt')
	#my_trainer.predict('corpus_scores\\10_opt_raw.txt')

if __name__ == "__main__":
	main()