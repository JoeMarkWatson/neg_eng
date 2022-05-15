import pandas as pd
import twint
import time
import datetime
import re
import pickle
import os
import matplotlib.pylab as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from scipy.stats import shapiro
from scipy.stats import normaltest
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


root_path = "/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/NOWn/"


def get_source_info(year):
    """returns df containing source files info, including url and title sentiment analysis"""
    i = 1
    d = []
    source_files = os.listdir(root_path + "NOW" + str(year) + "n/source_files")
    for f in source_files:
        if f != '.DS_Store':
            print(f)
            with open(root_path + "NOW" + str(year) + "n/source_files/" + f, "r", encoding='cp1252',
                      errors="replace") as file:
                for line in file:
                    fields = line.split("\t")
                    if len(fields) > 4:
                        if fields[0].isdigit():
                            if 'www.dailymail.co.uk' in fields[5]:
                                d.append(
                                    {
                                        'source_number': i,
                                        'text_id': fields[0],
                                        'year': year,
                                        'art_word_len': fields[1],
                                        'date': fields[2],
                                        'country': fields[3],
                                        'website': fields[4],
                                        'url': fields[5],
                                        'title': fields[6],
                                    }
                                )
                                i += 1
    return pd.DataFrame(d)


removes = ['<h', '<p', '\n', "' <h", "' <p", 'Advertisement <p', 'Advertisement <h', 'Share <p',
           'Share <h']  # for removing from sentences before applying Vader
tcm = ['RELATED ARTICLES <', 'Share this article <', 'Your details from Facebook will',
       'like it to be posted to Facebook', 'marketing and ads in line with our',
       'confirm this for your first post to Facebook', 'link your MailOnline account with',
       'will automatically post your comment and', 'time it is posted on MailOnline', 'link your MailOnline account',
       'first post to Facebook', 'comment will be posted to MailOnline', 'automatically post your MailOnline',
       'Share or comment on this']  # to remove 'share on facebook'-type requests - sometimes interspersed
       # with "@ @" symbols - without removing actual article sentences

def get_art_desc_senti(year, dd):
    """returns df containing art true word length and sentiment info"""
    year_dd = dd[dd['year'] == year]  # to reduce searching in 'if fields[0] in list(year_dd['text_id']):'
    i = 1
    d = []
    text_files = os.listdir(root_path + "NOW" + str(year) + "n/text_files")
    for f in text_files:
        if f != '.DS_Store':
            print(f)
            with open(root_path + "NOW" + str(year) + "n/text_files/" + f, "r", encoding='cp1252',
                      errors="replace") as file:  # fine to replace with '?' as raw NOW data uses this character in
                # place of characters such as '£'
                for line in file:
                    fields = line.split(" ", 1)  # maxsplit of 1
                    if len(fields) > 1:
                        fields[0] = re.sub("[^0-9]", "", fields[0])  # retain only numbers from fields[0]
                        if len(fields[0]) > 1:  # i.e., not NA and therefore .isdigit()
                            if fields[0] in list(year_dd['text_id']):
                                print("yes")
                                sentences = re.split("\. |\> |! ", fields[1])  # '?' not used given with open comment
                                sentences = [s for s in sentences if
                                             s not in removes and len(s) > 1 and not any(t in s for t in tcm)]
                                art_sentences_len = len(sentences)
                                if len(sentences) > 1:  # added for 2019 data to catch art with no text
                                    my_word_count, my_word_len_sum, neg_my_word_count = 0, 0, 0  # for whole art, not just by sentence
                                    art_vs_neg_comp_count, art_vs_clear_neg_comp_count = 0, 0
                                    for s in sentences:
                                        words_s = nltk.word_tokenize(s)
                                        words_s = [w.lower() for w in words_s]
                                        for ws in words_s:
                                            if len(re.findall(r'\w+', ws)) == 1:
                                                my_word_count += 1
                                                my_word_len_sum += len(ws)
                                                neg_my_word_count = neg_my_word_count + analyzer.polarity_scores(ws)['neg']
                                        vs = analyzer.polarity_scores(s)
                                        if vs['compound'] < 0:  # for prop of negative, neutral and positive sentences
                                            art_vs_neg_comp_count += 1
                                        if vs['compound'] <= -0.05:  # true vals follow https://github.com/cjhutto/vaderSentiment
                                            art_vs_clear_neg_comp_count += 1
                                    mean_word_len = my_word_len_sum/my_word_count
                                    prop_neg_words = neg_my_word_count/my_word_count
                                    d.append(
                                        {
                                            'source_number': i,
                                            'text_id': fields[0],
                                            'art_sentences_len': art_sentences_len,
                                            'my_word_count': my_word_count,
                                            'art_neg_word_prop': neg_my_word_count/my_word_count,
                                            'mean_my_word_len': mean_word_len,
                                            'mean_my_words_in_sen': my_word_len_sum/art_sentences_len,
                                            'art_neg_sent_prop': art_vs_neg_comp_count / art_sentences_len,
                                            'art_true_neg_sent_prop': art_vs_clear_neg_comp_count / art_sentences_len,
                                        }
                                    )
                                    i += 1
    return pd.DataFrame(d)


dd_2019 = get_source_info(2019)
dd_2020 = get_source_info(2020)
dd_2021 = get_source_info(2021)
dd_all = pd.concat([dd_2019, dd_2020, dd_2021])  # untested

dd2_2019 = get_art_desc_senti(2019, dd=dd_all)
dd2_2020 = get_art_desc_senti(2020, dd=dd_all)
dd2_2021 = get_art_desc_senti(2021, dd=dd_all)
dd2_all = pd.concat([dd_2019, dd_2020, dd_2021])  # untested

