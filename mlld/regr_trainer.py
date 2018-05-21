from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score
from .utils.data_reader import DataReader as dr
import sys, getopt, numpy as np

class Regression(object):

	def __init__(self, decimal_precision, debug):
		np.set_printoptions(threshold = np.inf)
		self.precision = decimal_precision
		self.debug = debug
		self.regr = LogisticRegression(class_weight = 'balanced')

	def train(self, features, labels):
		non_zero = np.count_nonzero(labels != 0)
		if self.debug:
			print('[INFO] ', non_zero, ' links present in training data, out of ', len(labels))
		self.regr.fit(features, labels)

	def predict(self, file):
		data = dr(file).read_data(precision = self.precision)
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
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)

		result = []
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.regr.predict(features[test])
			precision = precision_score(labels[test], preds, pos_label = 1, average = 'binary')
			if self.debug:
				print(classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))
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
					preds.extend(self.predict_from_proba(self.regr.predict_proba(feature), proba_tol))
					feature = [features[x]]
				else:
					feature.append(features[x])
			preds.extend(self.predict_from_proba(self.regr.predict_proba(feature), proba_tol))
			if self.debug:
				print(classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))
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

def main(argv):
	my_trainer = Regression(-1, debug = False)
	try:
		opts, args = getopt.getopt(argv,"d")
	except getopt.GetoptError:
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-d':
			my_trainer = Regression(-1, debug = True)

	#print(my_trainer.validate('corpus_scores\\v2_5_raw.txt'))
	print(my_trainer.validate_proba('corpus_scores\\v2_5_raw_inv.txt', 0.872))

if __name__ == "__main__":
   main(sys.argv[1:])