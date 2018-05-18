from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, precision_score
from data_reader import DataReader as dr
import numpy as np

class SVMTrainer(object):

	def __init__(self, decimal_precision, debug):
		np.set_printoptions(threshold = np.inf)
		
		self.precision = decimal_precision
		self.debug = debug
		self.svc = SVC(class_weight = 'balanced', probability = True)

	def scale_data(self, train_data):
		self.scaler = preprocessing.StandardScaler().fit(train_data)
		return self.scaler.transform(train_data)

	def train(self, features, labels):
		non_zero = np.count_nonzero(labels != 0)
		if self.debug:
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

		result = []
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.svc.predict(features[test])
			precision = precision_score(labels[test], preds, pos_label = 1, average = 'binary')
			if self.debug:
				print('[INFO] Precision: ', precision)
			result.append(precision)
		return result

	def validate_proba(self, file, proba_tol):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		
		result = []
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = []
			feature = [features[test[0]]]
			for x in test[1:]:
				if features[x - 1][-2] > features[x][-2]:
					print(self.svc.predict_proba(feature))
					print(self.predict_from_proba(self.svc.predict_proba(feature), proba_tol))
					break;
					preds.extend(self.predict_from_proba(self.svc.predict_proba(feature), proba_tol))
					feature = [features[x]]
				else:
					feature.append(features[x])
			preds.extend(self.predict_from_proba(self.svc.predict_proba(feature), proba_tol))
			#print(classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))
			result.append(precision_score(labels[test], preds, pos_label = 1, average = 'binary'))
		return result

	def predict_from_proba(self, proba, proba_tol):
		max = 0
		for i, val in enumerate(proba):
			if val[1] > max and val[1] >= proba_tol:
				max = val[1]
				idx = i

		result = np.zeros(len(proba), dtype = int)
		if max != 0:
			result[idx] = 1
		return result

def main():
	my_trainer = SVMTrainer(-1, debug = True)
	my_trainer.validate('corpus_scores\\v2_5_raw_inv.txt')
	#my_trainer.validate_proba('corpus_scores\\v2_5_raw_inv.txt', 0.8)
	#my_trainer.predict('corpus_scores\\10_opt_raw.txt')

if __name__ == "__main__":
	main()