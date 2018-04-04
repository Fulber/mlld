import xml.etree.ElementTree as ET
from utils.ranker import UtteranceRanker as UR

class ConvParser(object):

	def __init__(self, xml_file_name):
		self.file_name = xml_file_name
		self.doc = ET.parse(xml_file_name)
		self.dialog = self.doc.getroot()
		self.ur = UR()

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

	def prepareData(self, lang, depth):
		turns = self.get_turns()
		print('Ranking ', len(turns), 'turns from ' + self.file_name)
		
		ranks = []
		count = 0
		for i in range(len(turns) - depth):
			auth1 = turns[i]['nickname']
			utt1 = turns[i]['utterance']

			for j in range(depth):
				auth2 = turns[i + j + 1]['nickname']
				utt2 = turns[i + j + 1]['utterance']
				
				print('---[' + self.file_name, ':', utt1['genid'], '~', utt2['genid'], ']---')
				rank = self.ur.readerbench_all(utt1['text'], utt2['text'], lang)['similarityScores']
				rank['question'] = int(self.ur.is_question(utt1['text']))
				rank['authorReference'] = self.ur.author_reference(auth1, utt1['text'], auth2, utt2['text'])
				rank['link'] = int(utt1['genid'] == utt2['ref'])
				
				ranks.append(rank)
				print(rank)
			
			count += 1

		print('Succesfully ranked ', count, 'turns from ' + self.file_name)
		return ranks

def main():
	my_parser = ConvParser("corpus_chats\\1.xml")
	turns = my_parser.prepareData('English', 5)
	print(turns)

if __name__ == "__main__":
	main()