from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, make_scorer
from utils.data_reader import DataReader as dr
import sys, getopt, numpy as np

class RandomForest(object):

	def __init__(self, decimal_precision, debug):
		np.set_printoptions(threshold = np.inf)
		self.precision = decimal_precision
		self.debug = debug
		self.rfc = RandomForestClassifier(n_estimators = 20, max_depth = 20, criterion = 'entropy', random_state = 0, class_weight = 'balanced')
		#self.rfc = RandomForestClassifier(random_state = 0, n_estimators = 20, criterion = 'entropy', max_depth = 5)#experimental optimised

	def train(self, features, labels):
		non_zero = np.count_nonzero(labels != 0)
		if self.debug:
			print('[INFO] ', non_zero, ' links present in training data, out of ', len(labels))
		self.rfc.fit(features, labels)

	def predict(self, file):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 0:-1]
		labels = data[:, -1].astype(int)
		
		print('[INFO] ', np.count_nonzero(labels != 0), 'links present in training data, out of ', len(labels))

		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			print('[INFO] Round of testing')
			self.train(features[train], labels[train])
			
			preds = self.rfc.predict(features[test])
			print('[INFO] ', np.count_nonzero(preds != 0), ' links retrieved by model')
			
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
			preds = self.rfc.predict(features[test])
			precision = precision_score(labels[test], preds, pos_label = 1, average = 'binary')
			if self.debug:
				print('[INFO] ', np.count_nonzero(preds != 0), ' links retrieved by model')
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
					preds.extend(self.predict_from_proba(self.rfc.predict_proba(feature), proba_tol))
					feature = [features[x]]
				else:
					feature.append(features[x])
			preds.extend(self.predict_from_proba(self.rfc.predict_proba(feature), proba_tol))
			if self.debug:
				print('[INFO] ', np.count_nonzero(preds != 0), ' links retrieved by model')
				print('[INFO] Report', classification_report(labels[test], preds, target_names = ['class 0', 'class 1']))
			precision = precision_score(labels[test], preds, pos_label = 1, average = 'binary')
			result.append(precision)
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

	def tune_parameters(self, file):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		
		fT, ft, lT, lt = train_test_split(features, labels, test_size = 0.5, random_state = 0)
		parameters = [{'n_estimators': [1, 3, 5, 7, 10, 15, 20], 'max_depth': [5, 10, 20, 25, 30, 50,], 'criterion': ['gini', 'entropy']}]
		
		clf = GridSearchCV(RandomForestClassifier(random_state = 0, class_weight = 'balanced'), parameters, cv = 2, scoring = make_scorer(precision_score, pos_label = 1))
		clf.fit(fT, lT)
		print("-----\nBest parameters set found for Random Forest Classifier:\n-----")
		print(clf.best_params_)
		print(classification_report(lt, clf.predict(ft)))

def main(argv):
	my_trainer = RandomForest(-1, debug = False)
	try:
		opts, args = getopt.getopt(argv,"d")
	except getopt.GetoptError:
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-d':
			my_trainer = RandomForest(-1, debug = True)

	#print(my_trainer.validate('corpus_scores\\v2_5_raw_inv.txt'))
	print(my_trainer.validate('corpus_scores\\10_opt_raw.txt'))
	#my_trainer.tune_parameters('corpus_scores\\v2_5_raw_inv.txt')
	#my_trainer.tune_parameters('corpus_scores\\10_opt_raw.txt')
	#print(my_trainer.validate_proba('corpus_scores\\v2_5_raw_inv.txt', 0.47))

if __name__ == "__main__":
   main(sys.argv[1:])