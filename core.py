from os import listdir
from os.path import isfile, join
import json
from utils.parser import ConvParser

class Core(object):

	def __init__(self, corpus_dir):
		self.corpus_dir = corpus_dir

	def write_raw(self, file, ranks):
		with open(file, 'a+') as f:
			for rank in ranks:
				for _, v in rank.items():
					f.write(str(v) + ' ')
				f.write('\n')

	def write_json(self, file, ranks):
		with open(file, 'a+') as f:
			json.dump(ranks, f)

	def process_one(self, file_name):
		cp = ConvParser(file_name)
		ranks = cp.prepareData('English', 5)
		
		self.write_raw('data.txt', ranks)
		
		return ranks

	def process_all(self, start):
		corpus = [join(self.corpus_dir, f) for f in listdir(self.corpus_dir) if isfile(join(self.corpus_dir, f)) and f.endswith('.xml')]
		all_ranks = []
		
		for i in range(start, len(corpus)):
			ranks = self.process_one(corpus[i])
			all_ranks.append(ranks)
	
def main():
	core = Core("corpus_chats")
	core.process_all(6)

if __name__ == "__main__":
	main()