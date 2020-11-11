import re
from nltk.corpus import stopwords
import en_core_web_sm
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def visual_topTopics(model, ktopics, kwords, plotfile):
    top_words = [[word for word, _ in model.show_topic(topic_id, topn=50)] for topic_id in range(model.num_topics)]
    top_betas = [[beta for _, beta in model.show_topic(topic_id, topn=50)] for topic_id in range(model.num_topics)]

    ncols = math.ceil(math.sqrt(ktopics))
    nrows = math.ceil(ktopics * 1. / ncols)
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20, 15))
    for i in range(ktopics):
        ax = plt.subplot(gs[i])
        plt.barh(range(kwords), top_betas[i][:kwords], align='center', color='blue', ecolor='black')
        ax.invert_yaxis()
        plt.xticks(rotation=45)
        ax.set_yticks(range(kwords))
        ax.set_yticklabels(top_words[i][:kwords])
        plt.title("Topic " + str(i), fontsize=14)

    plt.savefig(plotfile)
