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

def main():
	my_reader = DataReader('corpus_scores\\10_opt_raw.txt')
	print(my_reader.read_data(-1))

if __name__ == "__main__":
	main()