import gensim
from gensim.models.wrappers import LdaMallet
from gensim.models.callbacks import PerplexityMetric,CoherenceMetric
import pandas as pd
import numpy as np
import math
import time
import os
import shutil
import json
from gensim.models import CoherenceModel
import logging, sys
import gc
import re
from nltk.corpus import stopwords
import spacy
import en_core_web_sm
from gensim.corpora.dictionary import Dictionary
from json import dumps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud

import utils



if __name__ == "__main__":
    #### PREPARING DATA ####
    print("Preparing data ...")

    # The folder where the data is
    datafile = '../breast_cancer_analysis/BreastCancer_Rawdata_TWs_unlabeled_predicted_rmSpam.csv'
    outputpath = 'outputs/TWs_unlabeled_predicted'

    df = pd.read_csv(datafile)
    # df.columns = ["index", "Unnamed: 0", "tweet_id", "timestamp", "text", "class"]
    df = df[df['text'].notna()]
    text_ids = df.tweet_id.to_list()

    # select one class
    suffix = "_predicted_rmSpam_class0_coher10ws"
    df = df[df['class'] == 0]

    # Preprocess Tweets and Tokenizes
    nlp = en_core_web_sm.load(disable=['parser', 'ner'])
    texts = [utils.preprocessing(x) for x in df['text']]
    # print(texts)

    # bag of words
    bow_dictionary = Dictionary(texts)
    bow_corpus = [bow_dictionary.doc2bow(tx) for tx in texts]

    # bigrams
    bigram_generator = gensim.models.Phrases(texts, min_count=20, threshold=50)  # higher threshold fewer phrases.
    bigram_model = gensim.models.phrases.Phraser(bigram_generator)
    # trigrams
    trigram_generator = gensim.models.Phrases(bigram_generator[texts], threshold=50)
    trigram_model = gensim.models.phrases.Phraser(trigram_generator)
    # bigram_model.save(f'{outputpath}/bigram_model{suffix}')
    # trigram_model.save(f'{outputpath}/trigram_model{suffix}')

    texts_ngrams = [trigram_model[bigram_model[tx]] for tx in texts]
    # print(texts_ngrams)

    id2token = gensim.corpora.Dictionary(texts)
    # id2token.save(f'{outputpath}/id2token{suffix}')
    texts_id_bow = [id2token.doc2bow(tx) for tx in texts]
    

    #### TRAIN LDA MODEL ####
    print("Training ...")
    num_topics_list = [10, 15, 20]
    # topn_list = [10, 20]
    alpha = 'auto'

    result_list = []
    ldamodel_list = {}

    for num_topics in num_topics_list:
        eta = None  # 'auto' # eta is beta in the lecture. A different naming system.
        print(f'For num_topics = {num_topics}, alpha = {alpha}, eta = {eta}')

        perplexity_logger = PerplexityMetric(corpus=texts_id_bow, logger='shell')
        coherence_cv_logger = CoherenceMetric(corpus=texts_id_bow, logger='shell', coherence='c_v',
                                              texts=texts)

        # training gensim LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus=texts_id_bow, num_topics=num_topics, id2word=id2token,
                                                   random_state=123, update_every=0, alpha=alpha, eta=eta, passes=50,
                                                   callbacks=[perplexity_logger, coherence_cv_logger])

        # calculate the perplexity on the training data
        perplexity_train = ldamodel.log_perplexity(texts_id_bow)
        print(f'\tFor training corpus, the perplexity is {perplexity_train}')

        # calculate the coherence on the training data with topn words
        coherence_model_train = CoherenceModel(model=ldamodel, texts=texts, dictionary=id2token,
                                               coherence='c_v', topn=10)
        coherence_train = coherence_model_train.get_coherence()
        print(f'\tFor training corpus, the coherence is {coherence_train} for topn=10')

        # storing data
        data_dict = dict()
        data_dict['num_topics'] = num_topics
        data_dict['alpha'] = alpha
        data_dict['perplexity_train'] = perplexity_train
        data_dict['coherence_train'] = coherence_train
        result_list.append(data_dict)

        ldamodel_list[num_topics] = ldamodel

    # the dataframe of the performance of the LDA model using different hyper-parameters
    df_performance = pd.DataFrame(result_list)
    df_performance.to_csv(f'{outputpath}/df_performance{suffix}.csv', index=False)

    #### VISUALIZATION ####
    print("Visualizing ... ")

    # plot the coherence score
    fig, ax = plt.subplots()
    xlabs = df_performance['num_topics']
    plt.plot(np.arange(len(xlabs)), df_performance['coherence_train'])
    plt.xticks(np.arange(len(xlabs)), xlabs, rotation=45)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{outputpath}/plot_lda_coher_train{suffix}.pdf")

    # find the best model
    best_paras = df_performance.iloc[df_performance['coherence_train'].idxmax()]
    best_model = ldamodel_list[best_paras['num_topics']]

    best_model.save(f"{outputpath}/model_ntopics{best_paras['num_topics']}{suffix}")

    # visualize the top words for top K topics
    utils.visual_topTopics(best_model, best_paras['num_topics'], 10, f"{outputpath}/barplot_lda_ntopics{best_paras['num_topics']}{suffix}.pdf")


