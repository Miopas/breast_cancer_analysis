'''
    Data: Oct, 2020
    Author: Yuan-Chi Yang
    Objective:
        This script convert the corpus into bag-of-word representation, for LDA topic modeling
    Reference:
        this script mostly follows https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
'''

import gensim
import pandas as pd
import spacy
#python3 -m spacy download en_core_web_sm
import time
import os
import shutil
from json import dumps
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

w = ['!',',','.','?','-s','-ly','</s>','s']
#newStopWords = ['stopWord1','stopWord2']
#filtered_words = [word for word in word_list if word not in stopwords.words('english')]


def texs_ngrams_func(texts, trigram_model, bigram_model):
    '''
        This function apply trained trigram_model and bigram_model on the texts
        Params:
            texts: the list of tweets.
            bigram_model: bigram model
            trigram_nodel: trigram model
        Return:
            the list of tweets, where the tweets are represented by bigrams and trigrams
    '''
    return [trigram_model[bigram_model[tweet]] for tweet in texts]


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def texts_lemmatized_func(texts_ngrams):
    '''
        This function generate the lemmatized texts using package spacy
        Params:
            texts_ngrams: the list of tweets that are represnted by bigrams and trigrams, output of function texs_ngrams_func
        Return:
            texts_lemmatized: the list of texts that are lemmatized, where only 'NOUN', 'ADJ', 'VERB', and 'ADV' remains
    '''

    texts_lemmatized = []
    for tweet in texts_ngrams:
        doc = nlp(" ".join(tweet))
        texts_lemmatized.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])

    return texts_lemmatized


def texts_id_bow_func(texts_lemmatized, id2token):
    '''
        Thus function expressed texts_lemmatized as bag-of-word using token id
        Params:
            texts_lemmatized: the list of texts that are lemmatized. The output of function texts_lemmatized_func
            id2token: the gensim dictionary object that map id to token
        Return:
            the list of documents that are expressed as bag-of-word using token id
    '''
    return [id2token.doc2bow(text) for text in texts_lemmatized]


def texts2texts_lemmatized(texts, trigram_model, bigram_model):
    '''
        This function convert texts directly to texts_lemmatized
        Params:
            texts: the list of tweets.
            bigram_model: bigram model
            trigram_nodel: trigram model
        Return:
            texts_lemmatized: the list of texts that are lemmatized
    '''
    texts_ngrams = texs_ngrams_func(texts, trigram_model, bigram_model)
    texts_lemmatized = texts_lemmatized_func(texts_ngrams)
    return texts_lemmatized


def print_both(to_print, file):
    '''
        This function print the expression both to the screen and to a file (log file)
        Params:
            to_print: the string to be printed
            file: the path to the log file
        Return:
            None
    '''
    print(to_print)
    print(to_print, file=file)


if __name__ == "__main__":
    start_time = time.localtime()

    # set-up the output folder
    work_folder = '/content/'
    current_time_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dir_name = f'{work_folder}/{current_time_file}-bow'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # the log file
    logfile = open(f'{dir_name}/logfile', 'w')

    # the data
    df_train = pd.read_csv('/content/data_reddit_processed.csv')
    #df_val = pd.read_csv('E:/phd in emory/BioNLP/hw9/val_data.csv')

    # first read training data
    #df_train_text = df_train.text.replace("breast"," ")
    #df_train_text = df_train.text.replace("cancer", " ")
    #print(df_train_text)
    texts = [x.split() for x in df_train.text.to_list()]
    tweet_ids = df_train.user_id.to_list()

    # using training data to train bigram_model and trigram_model
    ts = time.time()
    bigram_generator = gensim.models.Phrases(texts, min_count=20, threshold=50)  # higher threshold fewer phrases.
    trigram_generator = gensim.models.Phrases(bigram_generator[texts], threshold=50)
    bigram_model = gensim.models.phrases.Phraser(bigram_generator)
    trigram_model = gensim.models.phrases.Phraser(trigram_generator)
    bigram_model.save(f'{dir_name}/bigram_model')
    trigram_model.save(f'{dir_name}/trigram_model')
    te = time.time()
    print_both(f'It takes {te - ts} s to generate and save!\n', file=logfile)

    # represent the text into bigram and trigram
    texts_ngrams = texs_ngrams_func(texts, trigram_model, bigram_model)

    # lemmatize the text
    ts = time.time()
    texts_lemmatized = texts_lemmatized_func(texts_ngrams)
    #print(texts_lemmatized)
    newStopWords = ['breast','cancer','go','will','be','can','just','also','!',',','.','?','-s','-ly','</s>','s','’','%','would','could']
    #yinhao = []
    texts_lemmatized_new = []
    for word_list in texts_lemmatized:
      #print(word)
      texts_tmp = []
      for word in word_list:
        if word not in stopwords.words('english'):
          if word not in newStopWords:
          #if word == "’":
            texts_tmp.append(word)
            #print(word)
      texts_lemmatized_new.append(texts_tmp)

    #texts_lemmatized1 = [word for word in texts_lemmatized if word not in stopwords.words('english')]
    #newStopWords = ['breast','cancer','go','will','be','can','just','also','!',',','.','?','-s','-ly','</s>','s']
    #texts_lemmatized2 = [word for word in texts_lemmatized1 if word not in newStopWords]
    texts_lemmatized = texts_lemmatized_new
    print(texts_lemmatized)
    te = time.time()
    print_both(f'It takes {te - ts} s to lemmatize the text!\n', file=logfile)

    # create the mapping from token to id
    id2token = gensim.corpora.Dictionary(texts_lemmatized)
    id2token.save(f'{dir_name}/id2token')
    texts_id_bow = texts_id_bow_func(texts_lemmatized, id2token)

    # export the processed texts_id_bow and texts_lemmatized to a json file
    with open(f'{dir_name}/train_dict.json', 'w') as f:
        for i, tweet_id in enumerate(tweet_ids):
            data_dict = {'tweet_id': tweet_id,
                         'text_lemmatized': texts_lemmatized[i],
                         'text_id_bow': texts_id_bow[i]}
            f.write(dumps(data_dict) + '\n')

    # perform the same processing on the validation data
    """texts_val = [x.split() for x in df_val.text.to_list()]

    texts_lemmatized_val = texts2texts_lemmatized(texts_val, trigram_model, bigram_model)

    texts_id_bow_val = texts_id_bow_func(texts_lemmatized_val, id2token)

    tweet_ids_val = df_val.tweet_id.to_list()

    # export the processed texts_id_bow and texts_lemmatized to a json file
    with open(f'{dir_name}/val_dict.json', 'w') as f:
        for i, tweet_id in enumerate(tweet_ids_val):
            data_dict = {'tweet_id': tweet_id,
                         'text_lemmatized': texts_lemmatized_val[i],
                         'text_id_bow': texts_id_bow_val[i]}
            f.write(dumps(data_dict) + '\n')"""

    """shutil.copy(f'{work_folder}/generate_bow.py', dir_name)

    # the pbs file for running on BMI cluster
    shutil.copy(f'{work_folder}/cluster.pbs', dir_name)
    # the shell file for running on BMI cluster
    shutil.copy(f'{work_folder}/run_generate_bow.sh', dir_name)"""

    end_time = time.localtime()
    print('Script Running Summary:')
    print_both(f'\tstart at: {time.strftime("%c %Z", start_time)}', file=logfile)
    print_both(f'\tend at: {time.strftime("%c %Z", end_time)}', file=logfile)
    print_both(f'\tIt takes {time.mktime(end_time) - time.mktime(start_time)} s\n', file=logfile)

    print('Finished!!', file=logfile)
    logfile.close()
    print('Finished!!')
