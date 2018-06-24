from sklearn.metrics import classification_report, precision_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys

class DataReader(object):

	def __init__(self, filename):
		self.filename = filename
		np.set_printoptions(threshold = np.inf)

	def read_data(self, precision):
		with open(self.filename, 'r') as f:
			lines = f.readlines()
		
		if precision == -1:
			return np.array([x.strip().split(' ') for x in lines]).astype(float)
		else:
			return np.around(np.array([x.strip().split(' ') for x in lines]).astype(float), decimals = precision)

	def get_features_labels(self, precision, start):
		data = self.read(precision)
		return {
			'features': data[:, start:-1],
			'labels': data[:, -1].astype(int)
		}

	def write_to_file(self, filename, data):
		with open(filename, 'a+') as f:
			for el in data:
				for _, v in el.items():
					f.write(str(v) + ' ')
				f.write('\n')

	def split_data(self, precision, test_size, random_state):
		data = self.read_data(precision = precision)
		features = data[:, 2:-1]
		labels = data[:, -1].astype(int)
		a, b, c, d = train_test_split(features, labels, test_size = test_size, random_state = random_state)
		return a, b, c, d

	@staticmethod
	def predict_from_proba(proba, proba_tol):
		max = 0
		for i, val in enumerate(proba):
			if val[1] > max and val[1] >= proba_tol:
				max = val[1]
				idx = i

		result = np.zeros(len(proba), dtype = int)
		if max != 0:
			result[idx] = 1
		return result

	@staticmethod
	def print_report(debug, preds, labels):
		if debug:
			print('[INFO] ', np.count_nonzero(np.array(preds) == 1), ' links retrieved by model')
			print(classification_report(labels, preds, target_names = ['class 0', 'class 1']))

def main():
	my_reader = DataReader('corpus_scores\\10_opt_raw.txt')
	print(my_reader.read_data(-1))

if __name__ == "__main__":
	main()