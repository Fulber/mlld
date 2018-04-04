import requests
import sys

class UtteranceRanker(object):

	HEADERS = {
		'Accept': 'application/json',
		'Content-Type': 'application/json'
	}

	def __init__(self):
		self = self

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

	def is_question(self, utt):
		req = requests.post(
			"http://localhost:9000/?properties={\"annotators\":\"parse\", \"outputFormat\":\"json\"}",
			headers = self.HEADERS,
			data = utt
		)
		sentences = req.json().get('sentences')
		for sentence in sentences:
			tree = sentence.get('parse')
			if any(tag in tree for tag in ['SBARQ', 'SQ', 'SBAR']):
				return True
		return False

	def author_reference(self, auth1, utt1, auth2, utt2):
		rank = 0
		if auth1.lower() in utt2.lower():
			rank += 1
		if auth2.lower() in utt1.lower():
			rank += 1
		return rank

def main():
	my_ranker = UtteranceRanker()
	#response = my_ranker.readerbench_all("Let's think od some activities... and then decide which technology is better", "As we previously disccoused i think a combination of them should be perfect", 'English')
	response = my_ranker.author_reference("corina", "Mona is out because of her Internet connection! so let's wait for her!", "mona", "Sorry guys")
	print(response)

if __name__ == "__main__":
	main()