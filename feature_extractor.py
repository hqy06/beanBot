import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

text="Hello 3 I am happy"
def preprocessing(text):
	# to lower case
	text = text.lower()
	# remove numbers
	text = re.sub(r'\d+',' ', text)
	# remove punctuation
	text = text.translate(string.punctuation)
	# remove whitespaces
	text = text.strip()
	# Tokenization and remove stop words
	stop_words = set(stopwords.words('english')) 
	word_tokens = word_tokenize(text) 
	filtered_tokens = [w for w in word_tokens if not w in stop_words]
	# Stemming
	lemmatizer=WordNetLemmatizer()
	for w in filtered_tokens:
		w=lemmatizer.lemmatize(w)
	print(filtered_tokens)
	return filtered_tokens

def main():
	preprocessing(text)

if __name__ == '__main__':
	main()