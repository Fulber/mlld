from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, f1_score, make_scorer
from utils.data_reader import DataReader as dr
import sys, getopt, numpy as np

class Regression(object):

	def __init__(self, decimal_precision, debug, optimize, exp):
		np.set_printoptions(threshold = np.inf)
		self.precision = decimal_precision
		self.debug = debug
		self.regr = LogisticRegression(class_weight = 'balanced')
		if optimize:
			if exp:
				self.regr = LogisticRegression(C = 1000, tol = 1e-3, class_weight = 'balanced')#experimental optimized
			else:
				# precision and f1_score optimized
				self.regr = LogisticRegression(C = 100,  tol = 1e-3, class_weight = 'balanced')
		
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
			self.train(features[train], labels[train])
			preds = self.regr.predict(features[test])
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
			preds = self.regr.predict(features[test])
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
		parameters = [{'C': [1, 5, 10, 50, 100, 500, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
			'class_weight': ['balanced', None]}]
		
		clf = GridSearchCV(LogisticRegression(), parameters, cv = 2, scoring = make_scorer(precision_score, pos_label = 1))
		clf.fit(fT, lT)
		print("-----\nBest parameters set found for Logistic Regression tuning precision:\n-----")
		print(clf.best_params_)
		print(classification_report(lt, clf.predict(ft)))

	def kek(self, proba, proba_tol):
		file, exp_file = 'corpus_scores\\v2_5_raw_inv.txt', 'corpus_scores\\10_opt_raw.txt'
		data = dr(file).read_data(precision = self.precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		exp_data = dr(exp_file).read_data(precision = self.precision)
		exp_features = data[:, 2:-1]
		exp_labels = data[:, -1].astype(int)
		
		f1, f2, l1, l2 = train_test_split(features, labels, test_size = 0.5, random_state = 0)
		ef1, ef2, el1, el2 = train_test_split(exp_features, exp_labels, test_size = 0.5, random_state = 0)
		self.train(ef1, el1)
		if proba:
			preds = self.proba_util(f2, proba_tol)
			result = [precision_score(l2, preds, pos_label = 1)]
			dr.print_report(self.debug, preds, l2) #Print Classification Report
			
			self.train(ef2, el2)
			preds = self.proba_util(f1, proba_tol)
			result.append(precision_score(l1, preds, pos_label = 1))
			dr.print_report(self.debug, preds, l1) #Print Classification Report
		else:
			preds = self.regr.predict(f2)
			result = [precision_score(l2, preds, pos_label = 1)]
			dr.print_report(self.debug, preds, l2) #Print Classification Report
			
			self.train(ef2, el2)
			preds = self.regr.predict(f1)
			result.append(precision_score(l1, preds, pos_label = 1))
			dr.print_report(self.debug, preds, l1) #Print Classification Report
		return result

	def proba_util(self, features, proba_tol):
		preds = []
		feature = [features[0]]
		for i, x in enumerate(features[1:]):
			if features[i][-2] > x[-2]:
				preds.extend(dr.predict_from_proba(self.regr.predict_proba(feature), proba_tol))
				feature = [x]
			else:
				feature.append(x)
		preds.extend(dr.predict_from_proba(self.regr.predict_proba(feature), proba_tol))
		return preds

def main(argv):
	debug, proba, tune, optimize, exp, final = False, False, False, False, False, False
	proba_tol = 0.5 #0.7 precision; 0.5 f1_score
	file = 'corpus_scores\\v2_5_raw_inv.txt'
	try:
		opts, args = getopt.getopt(argv,"dpetof")
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
		elif opt == '-f':
			final = True

	my_trainer = Regression(-1, debug = debug, optimize = optimize, exp = exp)
	if final:
		print(my_trainer.kek(proba, proba_tol))
	else:
		if tune:
			my_trainer.tune_parameters(file)
		else:
			if proba:
				print(my_trainer.validate_proba(file, proba_tol))
			else:
				print(my_trainer.validate(file))

if __name__ == "__main__":
   main(sys.argv[1:])