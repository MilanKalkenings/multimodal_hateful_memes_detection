import pandas as pd
from nltk import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
from transformers import DistilBertTokenizer, DistilBertForTokenClassification
import re


class SentimentEngineer:
    """
    A class to manually extract multiple features from textual data.
    sentiment feature extraction inspired by:
    https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c, last access: 31.05.2021, 8:13pm
    """

    def __init__(self):
        """
        Constructor.
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # sst 2 = stanford sentiment treebank, this distilBERT is finetuned to perform sentiment analysis
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.distilbert = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        # "bad words"
        with open("../bad_words/bad_words.txt") as f:
            lines = f.readlines()

        # delete the "\n"s
        for i in range(len(lines)):
            lines[i] = lines[i][:-2]
        self.bad_words_list = lines

    def rel_bad_words_amount(self, text_data: pd.Series):
        def count(sequence: str):
            amount = 0
            sequence_split = sequence.split()  # approx
            for word in sequence_split:
                if word.lower() in self.bad_words_list:
                    amount += 1
            rel_amount = amount/len(sequence_split)
            return rel_amount

        bad_words_amount = text_data.apply(func=count)
        bad_words_amount.name = "rel_bad_words_amount"
        return bad_words_amount

    @staticmethod
    def textblob_sequence_sentiment(text_data: pd.Series):
        """
        Estimates the sequence sentiment using TextBlob.
        :param pd.Series text_data: a Series containing the sequences
        :return: a pd.Series containing the estimated sequence sentiments.
        """

        def sentiment(sequence: str):
            """
            Estimates the sentiment of one specific sequence using TextBlob.
            :param str sequence: a text sequence
            :return: the estimated sentiment
            """
            return TextBlob(sequence).sentiment[0]

        textblob_sequence_sent = text_data.apply(func=sentiment)
        textblob_sequence_sent.name = "tb_s_sent"
        return textblob_sequence_sent

    @staticmethod
    def textblob_sequence_subjectivity(text_data: pd.Series):
        """
        Estimates the sequence subjectivity using TextBlob.
        :param pd.Series text_data: a Series containing the sequences
        :return: the estimated subjectivity
        """

        def subjectivity(sequence: str):
            """
            Estimates the subjectivity of one specific sequence using TextBlob.
            :param str sequence: a text sequence
            :return: the estimated subjectivity
            """
            return TextBlob(sequence).sentiment[1]

        textblob_sequence_sub = text_data.apply(func=subjectivity)
        textblob_sequence_sub.name = "tb_s_sub"
        return textblob_sequence_sub

    def vader_sequence_sentiment(self, text_data: pd.Series):
        """
        Estimates the sequence sentiment using vader.
        :param pd.Series text_data: a Series containing the sequences
        :return: a pd.Series containing the estimated sequence sentiments.
        """
        analyzer = self.vader_analyzer

        def seq_sentiment(sequence: str):
            """
            Estimates the sentiment of one specific sequence using vader.
            :param str sequence: a text sequence
            :return: the estimated sentiment
            """
            return analyzer.polarity_scores(text=sequence)["compound"]

        vader_sequence_sent = text_data.apply(func=seq_sentiment)
        vader_sequence_sent.name = "vader_s_sent"
        return vader_sequence_sent

    def textblob_tokenwise_sentiment(self, text_data: pd.Series):
        """
        Estimates the token sentiments using TextBlob. Only the most negative and the most positive are stored.
        :param pd.Series text_data: a Series containing the sequences
        :return: the estimated extreme token sentiments
        """
        tweet_tokenizer = self.tweet_tokenizer

        def worst_sentiment(sequence: str):
            """
            Extracts the worst token sentiment.

            :param str sequence: a text sequence
            :return: the worst sentiment a token in the sequence achieves
            """
            tokens = tweet_tokenizer.tokenize(text=sequence)
            worst = 0
            for token in tokens:
                sentiment = TextBlob(token).sentiment[0]  # entry 0 contains sentiment/polarity, 1 contains subjectivity
                if sentiment < worst:
                    worst = sentiment
            return worst

        def best_sentiment(sequence: str):
            """
            Extracts the best token sentiment.

            :param str sequence: a text sequence
            :return: the best sentiment a token in the sequence achieves
            """
            tokens = tweet_tokenizer.tokenize(text=sequence)
            best = 0
            for token in tokens:
                sentiment = TextBlob(token).sentiment[0]
                if sentiment > best:
                    best = sentiment
            return best

        worst_textblob_token_sent = text_data.apply(func=worst_sentiment)
        worst_textblob_token_sent.name = "neg_tb_t_sent"
        best_textblob_token_sent = text_data.apply(func=best_sentiment)
        best_textblob_token_sent.name = "pos_tb_t_sent"
        return pd.concat([best_textblob_token_sent, worst_textblob_token_sent], axis=1)

    def extract_text_sentiments(self, df: pd.DataFrame):
        """
        Extracts all features this class can automatically extract.

        :param df:
        :return:
        """
        def combine(row: pd.Series):
            return torch.Tensor([row["vader_s_sent"], row["pos_tb_t_sent"], row["neg_tb_t_sent"], row["tb_s_sent"], row["tb_s_sub"], row["rel_bad_words_amount"]])

        text = df["text"]
        vader = self.vader_sequence_sentiment(text_data=text)
        tb_token = self.textblob_tokenwise_sentiment(text_data=text)
        tb_seq = self.textblob_sequence_sentiment(text_data=text)
        tb_sub = self.textblob_sequence_subjectivity(text_data=text)
        rel_bad_words_amount = self.rel_bad_words_amount(text_data=text)
        resulting_features = pd.concat([vader,
                                        tb_token,
                                        tb_seq,
                                        tb_sub,
                                        rel_bad_words_amount],
                                       axis=1)
        combined_resulting_features = resulting_features.apply(combine, axis=1)
        return combined_resulting_features
