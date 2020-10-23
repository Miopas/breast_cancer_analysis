from wordcloud import WordCloud
import sys
import pandas as pd
import string
import matplotlib.pyplot as plt
import matplotlib

import spacy
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
filter_tokens = ['breast', 'cancer', 'breastcancer', 'amp', 'im']

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if __name__ == '__main__':
    infile = sys.argv[1]
    df = pd.read_csv(infile)

    col_name = 'text'
    if col_name not in df.columns:
        col_name = 'Text'

    texts = []
    #for text in df[col_name]:
    for i, row in df.iterrows():
        if 'Class' in df.columns and row['Class'] == 0:
            continue
        text = row[col_name]
        texts.append(preprocess(str(text)))
    print('data size:{}'.format(len(texts)))
    all_text = ' '.join(texts)

    for t in filter_tokens:
        all_stopwords.add(t)
    wordcloud = WordCloud(stopwords=all_stopwords, width=1600, height=800, max_font_size=200, 
                            background_color="white", colormap=matplotlib.cm.inferno).generate(all_text)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Input:{}, Size:{}'.format(infile, len(texts)))
    #plt.show()
    plt.savefig(infile+'.png')
