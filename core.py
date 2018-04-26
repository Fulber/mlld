from os import listdir
from os.path import isfile, join
import json
from utils.parser import ConvParser

class Core(object):

	def __init__(self, corpus_dir, scores_dir, optimize):
		self.corpus_dir = corpus_dir
		self.scores_dir = scores_dir
		self.depth = 10
		self.optimize = optimize

	def write_raw(self, file, ranks):
		with open(file, 'a+') as f:
			for rank in ranks:
				for _, v in rank.items():
					f.write(str(v) + ' ')
				f.write('\n')

	def write_json(self, file, ranks):
		with open(file, 'a+') as f:
			json.dump(ranks, f)

	def process_one(self, file_name, index):
		cp = ConvParser(file_name)
		ranks = cp.prepareData(lang = 'English', depth = self.depth, optimize = True)
		filename = 'opt_raw_' if self.optimize else 'raw_'
		self.write_raw(join(self.scores_dir, str(self.depth),  filename + str(index) + '.txt'), ranks)
		return ranks

	def process_all(self, start):
		corpus = [join(self.corpus_dir, f) for f in listdir(self.corpus_dir) if isfile(join(self.corpus_dir, f)) and f.endswith('.xml')]
		all_ranks = []
		
		for i in range(start, len(corpus)):
			ranks = self.process_one(corpus[i], i)
			all_ranks = all_ranks + ranks
		
		filename = '_opt_raw.txt' if self.optimize else '_raw.txt'
		self.write_raw(join(self.scores_dir, str(self.depth) + filename), all_ranks)

def main():
	core = Core("corpus_chats", "corpus_scores", False)
	core.process_all(0)

if __name__ == "__main__":
	main()