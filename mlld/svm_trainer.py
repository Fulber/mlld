from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, f1_score, make_scorer
from utils.data_reader import DataReader as dr
import sys, getopt, numpy as np

class SVMTrainer(object):

	def __init__(self, decimal_precision, debug, optimize, exp):
		np.set_printoptions(threshold = np.inf)
		self.precision = decimal_precision
		self.debug = debug
		self.svc = SVC(class_weight = 'balanced', probability = True)
		if optimize:
			if exp:
				self.svc = SVC(class_weight = 'balanced', C = 10, gamma = 1e-3, kernel = 'rbf', shrinking = True)#experimental optimized
			else:
				# precision and f1_score optimized
				self.svc = SVC(class_weight = 'balanced', C = 1000, gamma = 1e-3, kernel = 'rbf', shrinking = True, probability = True)
				# f1_score optimized
				self.svc = SVC(class_weight = 'balanced', C = 10, gamma = 1e-3, kernel = 'rbf', shrinking = True, probability = True)

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
			self.train(features[train], labels[train])
			preds = self.svc.predict(features[test])
			print('[INFO] ', np.count_nonzero(preds != 0), ' links retrieved by model')
			print('[INFO] Succesfully predicted ', np.count_nonzero(preds == labels[test]), ' links out of ', len(labels[test]))

	def validate(self, file):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)

		result = []
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.svc.predict(features[test])
			result.append(precision_score(labels[test], preds, pos_label = 1, average = 'binary'))
			dr.print_report(self.debug, preds, labels[test]) #Print Classification Report
		return result

	def validate_proba(self, file, proba_tol):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		
		result = []
		k_fold = KFold(n_splits = 2)
		for train, test in k_fold.split(features):
			self.train(features[train], labels[train])
			preds = self.proba_util(features[test], proba_tol)
			result.append(precision_score(labels[test], preds, pos_label = 1))
			dr.print_report(self.debug, preds, labels[test]) #Print Classification Report
		return result

	def tune_parameters(self, file):
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		
		fT, ft, lT, lt = train_test_split(features, labels, test_size = 0.5, random_state = 0)
		parameters = [{'C': [10, 100, 500, 1000], 'gamma': [1e-3, 1e-4, 1e-5], 'kernel': ['rbf', 'poly'], 
			'shrinking': [True, False], 'class_weight': ['balanced']}]
		
		clf = GridSearchCV(SVC(), parameters, cv = 2, scoring = make_scorer(f1_score, pos_label = 1))
		clf.fit(fT, lT)
		print("-----\nBest parameters set found for SVM tuning precision:\n-----")
		print(clf.best_params_)
		print(classification_report(lt, clf.predict(ft)))

	def proba_util(self, features, proba_tol):
		preds = []
		feature = [features[0]]
		for i, x in enumerate(features[1:]):
			if features[i][-2] > x[-2]:
				preds.extend(dr.predict_from_proba(self.svc.predict_proba(feature), proba_tol))
				feature = [x]
			else:
				feature.append(x)
		preds.extend(dr.predict_from_proba(self.svc.predict_proba(feature), proba_tol))
		return preds

def main(argv):
	debug, proba, tune, optimize, exp = False, False, False, False, False
	file = 'corpus_scores\\v2_5_raw_inv.txt'
	proba_tol = 0.28 #0.28 precision; 0.25 f1_score
	try:
		opts, args = getopt.getopt(argv,"dpeto")
	except getopt.GetoptError:
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-d':
			debug = True
		elif opt == '-p':
			proba = True
		elif opt == '-e':
			file = 'corpus_scores\\10_opt_raw.txt'
			exp = True
		elif opt == '-t':
			tune = True
		elif opt == '-o':
			optimize = True

	my_trainer = SVMTrainer(-1, debug = debug, optimize = optimize, exp = exp)
	if tune:
		my_trainer.tune_parameters(file)
	else:
		if proba:
			print(my_trainer.validate_proba(file, proba_tol))
		else:
			print(my_trainer.validate(file))

if __name__ == "__main__":
   main(sys.argv[1:])