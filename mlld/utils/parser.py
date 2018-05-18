import xml.etree.ElementTree as ET
from .ranker import UtteranceRanker as UR

class ConvParser(object):

	def __init__(self, xml_file_name):
		self.file_name = xml_file_name
		self.doc = ET.parse(xml_file_name)
		self.dialog = self.doc.getroot()
		self.ur = UR()
		
		if any(x in xml_file_name for x in ['8', '9', '10']):
			self.time_format = '%H.%M.%S'
		else:
			self.time_format = '%H:%M:%S'

	def get_turns(self):
		turns = []
		body = self.dialog.find('Body')
		for turn in body.findall('Turn'):
			utterance = turn.find('Utterance')
			if utterance.text and utterance.text != 'joins the room' and utterance.text != 'leaves the room':
				turns.append({
					'nickname': turn.get('nickname'),
					'utterance': {
						'genid': utterance.get('genid'),
						'ref': utterance.get('ref'),
						'time': utterance.get('time'),
						'text': utterance.text
					}
				})
		return turns

	def prepareData(self, lang, depth, optimize):
		turns = self.get_turns()
		print('Ranking ', len(turns), 'turns from ' + self.file_name)
		
		ranks = []
		count = 0
		for i in range(depth, len(turns)):
			auth2 = turns[i]['nickname']
			utt2 = turns[i]['utterance']

			for k in range(1, depth + 1):
				#print('---[' + self.file_name, ':', utt1['genid'], '~', utt2['genid'], ']---')
				auth1 = turns[i - k]['nickname']
				utt1 = turns[i - k]['utterance']
				rank = {
					'id1': int(utt1['genid']),
					'id2': int(utt2['genid'])
				}
				rank.update(self.ur.readerbench_all(utt1['text'], utt2['text'], lang)['similarityScores'])
				
				rank['question'] = self.ur.is_question(utt1['text'])
				rank['answer'] = self.ur.is_answer_to_question(utt2['text'])
				#rank['similar_words'] = self.ur.similar_words_count(utt1['text'], utt2['text'])
				
				rank['author1'] = self.ur.author_in_answer(auth1, utt1['text'], auth2, utt2['text'])
				rank['author2'] = self.ur.author_in_query(auth1, utt1['text'], auth2, utt2['text'])
				rank['author3'] = self.ur.author_in_cont(auth1, utt1['text'], auth2, utt2['text'])
				
				rank['distance1'] = self.ur.distance_in_queries(int(utt1['genid']), int(utt2['genid']))
				rank['distance2'] = self.ur.distance_in_times(utt1['time'], utt2['time'], self.time_format)
				
				rank['link'] = int(utt1['genid'] == utt2['ref'])
				ranks.append(rank)
				#print(rank)
			count += 1

		print('Succesfully ranked ', count, 'turns from ' + self.file_name)
		return ranks

def main():
	my_parser = ConvParser("corpus_chats\\1.xml")
	turns = my_parser.prepareData('English', 5, False)
	print(turns)

if __name__ == "__main__":
	main()