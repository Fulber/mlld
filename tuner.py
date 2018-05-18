from mlld.regr_trainer import Regression
from mlld.rfc_trainer import RandomForest
from mlld.ada_boost_trainer import AdaBoost
import sys, getopt

class Tuner(object):

	def __init__(self, algorithm, proba_tol, precision):
		self.proba_tol = proba_tol
		if algorithm == 'regr':
			self.alg = Regression(precision, debug = True)
		elif algorithm == 'rfc':
			self.alg = RandomForest(precision, debug = True)
		elif algorithm == 'ada':
			self.alg = AdaBoost(precision, debug = True)

	def average(self, arr):
		return sum(arr) / float(len(arr))

	def tune_proba(self, step):
		avg = [0]
		while self.proba_tol < 0.9:
			res = self.alg.validate_proba('corpus_scores\\v2_5_raw_inv.txt', self.proba_tol)
			if (self.average(res) > self.average(avg)):
				avg = res
				tol = self.proba_tol
			self.proba_tol = self.proba_tol + step
		print(tol, ' ', avg)

	def tune_parameters(self, arr):
		print('ok')

def main(argv):
	algorithm, proba_tol, step = '', 0.0, 0.1
	try:
		opts, args = getopt.getopt(argv,"ha:t:s:",["alg=", "tol=", "step="])
	except getopt.GetoptError:
		print('tuner.py -a <algorithm> -t <tolerance> -s <step>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('tuner.py -a <algorithm> -t <tolerance> -s <step>')
			sys.exit()
		elif opt in ("-a", "--alg"):
			algorithm = arg
		elif opt in ("-t", "--tol"):
			proba_tol = float(arg)
		elif opt in ("-s", "--step"):
			step = float(arg)

	tuner = Tuner(algorithm, proba_tol, -1)
	tuner.tune_proba(step)

if __name__ == "__main__":
   main(sys.argv[1:])