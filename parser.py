import xml.etree.ElementTree as ET

class ConvParser(object):

	def __init__(self, xml_file_name):
		self.doc = ET.parse(xml_file_name)
		self.dialog = self.doc.getroot()

	def get_turns(self):
		turns = []
		body = self.dialog.find('Body')
		for turn in body.findall('Turn'):
			utterance = turn.find('Utterance')
			if utterance.text != 'joins the room' and utterance.text != 'leaves the room':
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

def main():
	my_parser = ConvParser("corpus_chats\\1.xml")
	turns = my_parser.get_turns()
	print(turns)

if __name__ == "__main__":
	main()