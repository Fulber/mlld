from datetime import datetime
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
			if any(tag in tree for tag in ['SBARQ', 'SQ', 'SBAR', 'SINV']):
				return 1
		return 0

	def is_answer_to_question(self, utt):
		#TODO: check for possible answers: yes/no/ok/sure/agree/right/wrong/
		if any(x in ['yes', 'no', 'ok', 'sure', 'agree'] for x in utt.split()):
			return 1
		return 0

	def similar_words_count(self, utt1, utt2):
		return len(set(utt1.split()) & set(utt2.split()))

	def author_reference(self, auth1, utt1, auth2, utt2):
		rank = 0.0
		if auth1.lower() in utt2.lower():
			rank += 0.5
		if auth2.lower() in utt1.lower():
			rank += 0.5
		return rank

	def author_in_answer(self, auth1, utt1, auth2, utt2):
		if auth1.lower() in utt2.lower():
			return 1
		return 0

	def author_in_query(self, auth1, utt1, auth2, utt2):
		if auth2.lower() in utt1.lower():
			return 1
		return 0

	def author_in_cont(self, auth1, utt1, auth2, utt2):
		if auth1.lower() == auth2.lower():
			return 1
		return 0

	def distance_in_queries(self, id1, id2):
		return id2 - id1

	def distance_in_times(self, time1, time2, time_format):
		diff = datetime.strptime(time2, time_format) - datetime.strptime(time1, time_format)
		return diff.total_seconds()

def main():
	my_ranker = UtteranceRanker()
	#response = my_ranker.readerbench_all("Let's think od some activities... and then decide which technology is better", "As we previously disccoused i think a combination of them should be perfect", 'English')
	#response = my_ranker.author_in_cont("corina", "Mona is out because of her Internet connection! so let's wait for her!", "corina", "Sorry guys")
	response = my_ranker.is_answer_to_question("i dont know you")
	print(response)

if __name__ == "__main__":
	main()