import re
from nltk.corpus import stopwords
import en_core_web_sm

nlp = en_core_web_sm.load(disable=['parser', 'ner'])

def preprocessing(raw_text):
    # lowercasing, stemming and removing stop words
    sw = stopwords.words('english')
    words = []
    for w in raw_text.lower().split():
        # remove stop words
        if not w in sw:
            # remove non-ascii characters
            s = re.sub(r'[^\x00-\x7f]', r'', w)
            ss = re.sub(r'https?:\/\/.*', r'', s)
            if ss != '':
                words.append(ss)
    doc = nlp(" ".join(words))
    words_lemmatized = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    # remove some specific words, like "breast", "cancer", "#"
    words_lemmatized2 = [x for x in words_lemmatized if (x not in ["#", "-", "breast", "cancer", "breastcancer"]) and (len(x) > 1)]
    return words_lemmatized2