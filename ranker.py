import requests
import sys

class UtteranceRanker(object):

	HEADERS = {
		'Accept': 'application/json',
		'Content-Type': 'application/json'
	}

	def __init__(self, output_file_name):
		self.output = output_file_name

	def question():
		

	def readerbench_all(self, utt1, utt2, lang):
		req = requests.post(
			"http://readerbench.com/api/text-similarities",
			headers = self.HEADERS,
			json = {
				'text1': utt1,
				'text2': utt2,
				'language': lang,
				'pos-tagging': 'true',
				'lda': 'TASA',
				'lsa': 'TASA',
				'w2v': 'TASA'
			}
		)
		return req.json().get('data')

	def readerbench_one(self, utt1, utt2, lang, model, corpus):
		req = requests.post(
			"http://readerbench.com/api/text-similarity",
			headers = self.HEADERS,
			json = {
				'text1': utt1,
				'text2': utt2,
				'language': lang,
				'model': model,
				'corpus': corpus
			}
		)
		return req.json().get('data')

def main():
	my_ranker = UtteranceRanker("test.txt")
	response = my_ranker.readerbench_all("Let's think od some activities... and then decide which technology is better", "As we previously disccoused i think a combination of them should be perfect", 'English')
	print(response)

if __name__ == "__main__":
	main()