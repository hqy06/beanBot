"""
Generate synonyms of given words using WordNet from Princeton University.
------
About WordNet: https://wordnet.princeton.edu/
Coded by Fiona (@hqy06)
------
Which words for stemming:
1. https://www.engineering.uwaterloo.ca/~lgolab/social-media-mining.pdf
2. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5529738/
------
Sample usage is presented in main.
"""
# ===============================================
# Import Libraries
# ===============================================
from nltk.corpus import wordnet
from datetime import datetime


# ===============================================
# Global Varaibles
# ===============================================
STEM = "stem.txt"

# ===============================================
# Utilities
# ===============================================


def load_wordnet():
    """download/update WordNet"""
    import nltk
    nltk.download('wordnet')


def get_stems(file_path):
    """return the stems stored in the given file.
    ---
    copy and paste and a bit modification from my ai4good repo"""
    words = open(file_path, encoding='utf-8').read().strip().split('\n')
    return words


def save_generated_words(words):
    """save a list words in text file."""
    file_name = 'synonyms {}.txt'.format(
        datetime.now().strftime('%Y%m%d-%H%M'))
    a_file = open(file_name, 'w')
    text = '\n'.join(words)
    a_file.write(text)
    a_file.close()


def generate_synonyms(words, remove_duplicate=False):
    syn_dict = generate_syn_dict(words)
    synonyms = []
    for w in syn_dict.keys():
        synonyms += syn_dict[w]
    if remove_duplicate:
        synonyms = remove_list_duplicate(synonyms)
    return synonyms


def remove_list_duplicate(a_list):
    unique = []
    for item in a_list:
        if item not in unique:
            unique.append(item)
    return unique


def generate_syn_dict(words):  # TODO: test me!
    """return synonyms of words in dictionary format"""
    result = {}
    for w in words:
        result[w] = _generate_syn(w)
    return result


def _generate_syn(word):
    """return the synonyms of given word."""
    synonyms = []
    synset = wordnet.synsets(word)
    for s in synset:
        for l in s.lemmas():
            synonyms.append(l.name())
    return synonyms


# ===============================================
# Main
# ===============================================


def main():
    load_wordnet()
    words = get_stems(STEM)
    synonyms = generate_synonyms(words, remove_duplicate=True)
    save_generated_words(synonyms)


if __name__ == "__main__":
    main()
