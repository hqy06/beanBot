"""Interactive script that explores stuffs I use for word embedding."""

# ==================== explore WordNet
# %%
from nltk.corpus import wordnet
import nltk
from importlib import reload
import synGen
nltk.download('wordnet')    # download

dir(wordnet)    # methods in WordNet

# %%
# synset
w = "golden"
w = "sleep"
w_synset = wordnet.synsets(w)
w_synset

# lexical relation
synonyms, antonyms = [], []
for syn in w_synset:
    for l in syn.lemmas():
        synonyms.append(l.name())
        ant = l.antonyms()
        if ant:    # if antonyms exists!
            antonyms.append(ant[0].name())
synonyms, antonyms

# hypernym
for syn in w_synset:
    for h in syn.hyponyms():
        print(h.lemma_names()[0])


# ==================== explore WordNet
# %%
reload(synGen)
synGen.get_stems(synGen.STEM)

# %%
reload(synGen)
words = ['depress', 'stress', 'pain', 'homework']

synGen._generate_syn_words('homework')

synGen.generate_synonyms_dict(words)

result = synGen.generate_synonyms(words, True)

# ===================== save the result!
# %%
reload(synGen)
synGen.save_generated_words(result)
