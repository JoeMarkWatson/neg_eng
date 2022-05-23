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
                                    mean_word_len = my_word_len_sum / my_word_count
                                    d.append(
                                        {
                                            'source_number': i,
                                            'text_id': fields[0],
                                            'art_sentences_len': art_sentences_len,
                                            'my_word_count': my_word_count,
                                            'art_neg_word_prop': neg_my_word_count / my_word_count,
                                            'mean_my_word_len': mean_word_len,
                                            'mean_my_words_in_sen': my_word_len_sum / art_sentences_len,
                                            'art_neg_sent_prop': art_vs_neg_comp_count / art_sentences_len,
                                            'art_true_neg_sent_prop': art_vs_clear_neg_comp_count / art_sentences_len,
                                        }
                                    )
                                    i += 1
    return pd.DataFrame(d)


def tweets_search(mdd):  # mdd for merged_data_frame
    """scrapes all tweets featuring any URL in the dataframe, mdd"""
    tweet_text_dfs = []
    tweet_overview = []
    start_1000_time = datetime.datetime.now()
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
    return tweet_overview, tweet_text_dfs


# ur 'extracting art text for art.s in mdds, to simplify vectorising process in tfidf model loop'
# should go above scraping functions if it can


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
mdd = mdd[~mdd['url'].str.contains('/search.html')]  # remove url links that are site-wide keyword searches
mdd = mdd[~mdd['url'].str.contains('/index')]  # remove url links that show all trending news stories about a topic
mdd = mdd[~((mdd['url'].str.contains('books/article') == True) & (mdd['url'].str.contains('/amp/') == True))]  # remove
# url links to blurbs of one or multiple books in a book category, presented without additional comment.

mdd_2020_21 = mdd[mdd['year'] > 2019]  # this is a temp line - remove after merging
tweet_overview, tweet_text_dfs = tweets_search(mdd_2020_21)  # mdd_2020_21 gets changed to mdd after merging

len(tweet_text_dfs)
len(tweet_overview)

open_file = open(
    "/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_text_dfsALL.pkl", "wb")
pickle.dump(tweet_text_dfs, open_file)
open_file.close()
open_file = open(
    "/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_overviewALL.pkl", "wb")
pickle.dump(tweet_overview, open_file)
open_file.close()

# UR NEXT MERGING WITH 2019 TWEETS
# AND THEN UR PICKLING THAT, AND ALTERING THE ABOVE


# ok, 19 May 2022, when ur restarting on this, load the mdd file, load the tweet_text_dfs and tweet_overview file
# . Then do whatever merging u need to do with the existing 2019 files. Then save the output of that to a new pickle.
# . And then tidy this script as if everything was scraped in 1 go.
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/art_info_1.pkl", "rb") as f:
    mdd = pickle.load(f)
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_text_dfsALL.pkl", "rb") as f:
    tweet_text_dfs = pickle.load(f)
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_overviewALL.pkl", "rb") as f:
    tweet_overview = pickle.load(f)


# 22 May
todf = pd.DataFrame(tweet_overview, columns=['text_id', 'n_shares'])
mdds = pd.merge(mdd, todf, on='text_id')  # merge todf with mdd
mdds['n_shares'].sum()  # 25,327


# extracting art text for art.s in mdds, to simplify vectorising process in tfidf model loop

def extract_art_text(year):
    rel_art_texts = []
    text_files = os.listdir(root_path + "NOW" + str(year) + "n/text_files")
    for m in text_files:  # text_files and root_path stay same as above
        if m != '.DS_Store':
            print(m)
            with open(root_path + 'NOW' + str(year) + "n/text_files/" + m, "r", encoding='cp1252', errors="replace") as file:
                for line in file:
                    fields = line.split(" ", 1)  # maxsplit of 1
                    if len(fields) > 1:
                        fields[0] = re.sub("[^0-9]", "", fields[0])  # retain only numbers from fields[0]
                        if len(fields[0]) > 1:  # i.e., not NA and therefore .isdigit()
                            if fields[0] in list(mdd['text_id']):
                                sentences = re.split("\. |\> |! ", fields[1])
                                sentences = [s for s in sentences if
                                             s not in removes and len(s) > 1 and not any(t in s for t in tcm)]
                                rel_art_texts.append([fields[0], '. '.join(sentences)])  # return to a conv sentence
                                # structure, even though punctuation currently removed during tf-idf pre-processing
    return rel_art_texts

rel_art_texts_2019 = extract_art_text(2019)  # this is what you keep in script, but note that ur current mdd doesn't support
rel_art_texts_2020 = extract_art_text(2020)
rel_art_texts_2021 = extract_art_text(2021)


# wrangle data from tweets sharing DM articles

def wrangle_tweet_text(tweet_texts_dfs_input):
    """function to wrangle data from tweets sharing DM articles, which gives valence info following
    above article-focused manipulation"""
    d = []
    for ttd in tweet_texts_dfs_input:
        tweets = ttd[1]
        my_word_count, my_word_len_sum, neg_my_word_count, t_vs_maj_neg_words_count, t_vs_true_neg_count, \
        t_retweets, t_retweets_b, t_likes_b, t_replies_b, = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for index, row in tweets.iterrows():  # note poss shortcomings of iterrows: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            vs = analyzer.polarity_scores(row['tweet'])
            if vs['compound'] <= -0.05:  # true vals follow https://github.com/cjhutto/vaderSentiment
                t_vs_true_neg_count += 1
            t_retweets = t_retweets + row['nretweets']
            words_t = nltk.word_tokenize(row['tweet'])
            words_t = [w.lower() for w in words_t]
            for ws in words_t:
                if len(re.findall(r'\w+', ws)) == 1:
                    my_word_count += 1
                    my_word_len_sum += len(ws)
                    neg_my_word_count = neg_my_word_count + analyzer.polarity_scores(ws)['neg']
            if neg_my_word_count/my_word_count >= 0.5:  # true vals follow https://github.com/cjhutto/vaderSentiment
                t_vs_maj_neg_words_count += 1
            if row['nretweets'] > 0:
                t_retweets_b += 1  # b for binary
            if row['nlikes'] > 0:
                t_likes_b += 1
            if row['nreplies'] > 0:
                t_replies_b += 1
        d.append(
            {
                'text_id': ttd[0],
                't_vs_true_neg_prop': t_vs_true_neg_count / len(tweets),
                't_vs_maj_neg_words_prop': t_vs_maj_neg_words_count / len(tweets),
                'n_tweets_retweeted_tot': t_retweets,
                'n_tweets_retweeted_b_tot': t_retweets_b,
                'n_tweets_retweeted_b_prop': t_retweets_b / len(tweets),
                # This binary measure downwards not thrown off by presence of 1 high-follower sharer and avoids big skew
                'n_tweets_liked_b_prop': t_likes_b / len(tweets),
                'n_tweets_replies_b_prop': t_replies_b / len(tweets),
            }
        )
    return pd.DataFrame(d)

with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_text_dfs2019.pkl", "rb") as f:
    tweet_text_dfs_2019 = pickle.load(f)
dd = wrangle_tweet_text(tweet_text_dfs)  # leave only this
dd2 = wrangle_tweet_text(tweet_text_dfs_2019)  # remove this after
dd = pd.concat([dd, dd2], ignore_index=True)  # remove this after
# merge dd and mddsl
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/all_senti_shares_count2019.pkl",
          "rb") as f:
    mdds_2019 = pickle.load(f)  # remove this after
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/art_info_1.pkl", "rb") as f:
    mdd = pickle.load(f)  # remove this after
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_overviewALL.pkl", "rb") as f:
    tweet_overview = pickle.load(f)  # remove this after
todf = pd.DataFrame(tweet_overview, columns=['text_id', 'n_shares'])  # remove this after
mdds = pd.merge(mdd, todf, on='text_id')  # merge todf with mdd  # remove this after
mdds = pd.concat([mdds, mdds_2019], ignore_index=True)  # remove this after

mddsl = mdds[mdds['art_sentences_len'] > 9]  # remove art.s where less than 10 sentences  # repeated below as needed after pickle load
mddslrt = pd.merge(mddsl, dd, on="text_id", how='left')  # to keep rows where no shares
mddslrt = mddslrt[mddslrt['n_shares'] != 0]  # remove tweets where no retweets
mddslrt['mean_rt'] = mddslrt['n_tweets_retweeted_tot'] / mddslrt['n_shares']


#_____________________________________________________

# Creating pickles
# pickle mdds
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/mdds_19_20_21.pkl", "wb")
pickle.dump(mdds, open_file)
open_file.close()
# pickle mddslrt
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/mddslrt_19_20_21.pkl", "wb")
pickle.dump(mddslrt, open_file)
open_file.close()
# pickle tweet_text_dfs
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_text_dfs_19_20_21.pkl", "wb")
pickle.dump(tweet_text_dfs, open_file)
open_file.close()
# pickle rel_art_texts_2019
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2019_dup.pkl", "wb")
pickle.dump(rel_art_texts_2019, open_file)
open_file.close()
# pickle rel_art_texts_2020
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2020.pkl", "wb")
pickle.dump(rel_art_texts_2020, open_file)
open_file.close()
# pickle rel_art_texts_2021
open_file = open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2021.pkl", "wb")
pickle.dump(rel_art_texts_2021, open_file)
open_file.close()

# Loading pickles
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/mdds_19_20_21.pkl", "rb") as f:
    mdds = pickle.load(f)
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/mddslrt_19_20_21.pkl", "rb") as f:
    mddslrt = pickle.load(f)
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/tweet_text_dfs_19_20_21.pkl", "rb") as f:
    tweet_text_dfs = pickle.load(f)  # not currently used in analysis below
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2019.pkl", "rb") as f:
    rel_art_texts_2019 = pickle.load(f)  # not currently used in analysis below  # original save loc from 2019 only script
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2020.pkl", "rb") as f:
    rel_art_texts2020 = pickle.load(f)  # not currently used in analysis below
with open("/Users/joewatson/Desktop/JBS work/Psychometrician_position/fun_proj_attempts/rel_art_texts2021.pkl", "rb") as f:
    rel_art_texts2021 = pickle.load(f)  # not currently used in analysis below

# when ur next on this:
# 1. check ur pickles load and are rational objects
# 2. tidy up all the above, so it can all be produced from this script

