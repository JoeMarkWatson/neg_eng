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

# DEL THIS LINE AFTER FIN DRAFTING - note that the below fun draws from 169 onwards on neg_eng_2019...
def tweets_search(mdd):  # mdd for merged_data_frame
    """scrapes all tweets featuring any URL in the dataframe, mdd"""
    tweet_text_dfs = []
    tweet_overview = []
    i = 1  # base counter
    r_i = 1  # 1000-repeat counter
    for url in mdd['url']:
        print(str(i) + ": " + url)
        i += 1
        c = twint.Config()
        c.Limit = 10000  # to avoid over-calling - where 10,000 reached then can re-call later
        c.Lang = "en"
        text_id = mdd[mdd['url'] == url]['text_id'].item()
        url = re.sub("http://|https://", "", url)
        if url.startswith('www.'):
            url = url[4:]
        url = url.rstrip('/')  # strip all forward slashes from rhs
        # url = url.rstrip('.html|htm')  # not yet trialled  # YOU ARE REMOVING END OF SEARCH STRING WHICH MESSES UP TWITTER
        url = url.rstrip('html')  # rstrip removes any characters given as arg, regardless of their order or
        # quantity in string, and regardless of if one of the char.s is missing from end string
        url = url.rstrip('.')
        if 'html?' in url:  # removes part of html string that does not feature in Twitter shares
            url = url.split(".html?")[0]
        c.Search = url  # note that c.Search trialling shows no case effect
        # c.Search = "\"#HelptoGrow\""  # https://github.com/twintproject/twint/issues/231
        c.Pandas = True  # https://github.com/twintproject/twint/wiki/Pandas-integration
        twint_ran = False
        a = 5
        while twint_ran == False:
            try:
                if ((datetime.datetime.now() - start_1000_time).seconds < (
                        (60 * 60) - 5)) and r_i <= 999:  # 5s shy of 1h
                    print("r_i = " + str(r_i) + ", time so far for this 1000 = " +
                          str((round((datetime.datetime.now() - start_1000_time).seconds / 60, 2))) + " minutes")
                    twint.run.Search(c)
                    r_i += 1  # add one to 1000-repeat counter
                    twint_ran = True
                if ((datetime.datetime.now() - start_1000_time).seconds > (
                        (60 * 60) - 5)) and r_i > 999:  # where slow scraping
                    print("r_i = " + str(r_i) + ", time so far for this 1000 = " +
                          str((round((datetime.datetime.now() - start_1000_time).seconds / 60, 2))) + " minutes")
                    print("resetting timer\n")
                    start_1000_time = datetime.datetime.now()  # reset timer
                    twint.run.Search(c)
                    r_i = 2  # i.e., 1 + 1  # reset 1000-repeat counter
                    twint_ran = True
                if ((datetime.datetime.now() - start_1000_time).seconds < (
                        (60 * 60) - 5)) and r_i > 999:  # where fast scraping
                    hour_from_start = start_1000_time + datetime.timedelta(hours=1)
                    print("r_i = " + str(r_i) + ", time so far for this 1000 = " +
                          str((round((datetime.datetime.now() - start_1000_time).seconds / 60, 2))) + " minutes")
                    print("pausing until " + str(hour_from_start) + '\n')
                    time.sleep((hour_from_start - datetime.datetime.now()).seconds + 5)  # pause until end of hour + 5s
                    start_1000_time = datetime.datetime.now()  # reset timer
                    twint.run.Search(c)
                    r_i = 2  # i.e., 1 + 1  # reset 1000-repeat counter
                    twint_ran = True
            except:  # 'except' to cover error in above, including: assumption that 1000 requests can be made per hour; and,
                # cases where last few instances of a slow scraping 1000 were fast (causing limit breach on next 1000)
                print("twint error, pausing for", str(a), "minutes")
                time.sleep(60 * a)  # sleep for N mins
                a += 30  # add 30 minutes to the sleep
                if a >= 65:
                    r_i = 1
                    # twint_ran = False  # stays same as no scrape made
                    if a > 65:  # once 65 minutes sleep has been attempted (and a is 95)
                        break  # as clearly twint is not functioning or some other part of the script is incorrect
        Tweets_df = twint.storage.panda.Tweets_df
        tweet_overview.append([text_id, len(Tweets_df)])
        if len(Tweets_df) > 0:
            Tweets_df.drop(['search', 'translate', 'trans_src', 'trans_dest'], axis=1, inplace=True)
            tweet_text_dfs.append(
                [text_id, Tweets_df])  # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas
            # -dataframe-then-filling-it - why you are appending df.s to list
        print("_______________________________________ \n_______________________________________ \n")


# extract source file info
dd_2019 = get_source_info(2019)
dd_2020 = get_source_info(2020)
dd_2021 = get_source_info(2021)
dd_all = pd.concat([dd_2019, dd_2020, dd_2021])

# extract text file info
dd2_2019 = get_art_desc_senti(2019, dd=dd_all)
dd2_2020 = get_art_desc_senti(2020, dd=dd_all)
dd2_2021 = get_art_desc_senti(2021, dd=dd_all)
dd2_all = pd.concat([dd2_2019, dd2_2020, dd2_2021])

# inner join dd_all and dd2_all into mdd
mdd = pd.merge(dd_all, dd2_all, on='text_id')  # dd2 only attempted senti for art.s in dd, so len(dd) == len(dd2)
mdd = mdd.drop_duplicates(subset='url', keep="first")  # remove any url duplicates
mdd = mdd[mdd['url'] != 'https://www.dailymail.co.uk/home/index.rss%20']  # remove erroneous url (2020) before scraping

# save pickle mdd
open_file = open(
    "/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/art_info_1.pkl", "wb")
pickle.dump(mdd, open_file)
open_file.close()  # end mdds pickle

# load pickle mdd
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/art_info_1.pkl", "rb") as f:
    mdd = pickle.load(f)
mdd.drop(['source_number_x', 'source_number_y'], axis=1, inplace=True)  # drop source_number column names

# WHEN U FINISHED 17 MAY YOU JUST HAD DONE PICKLINEG - THAT GOT SAVED

