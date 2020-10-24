import numpy as numpy
import pandas as pd

def charcounts(x):
	s = x.split()
	x = ''.join(s)
	return len(x)

def wordcounts(x):
	length = len(str(x).split())
	return length

def avg_wordlength(x):
	count = charcounts(x)/wordcounts(x)
	return count

def basic_features(df):
    if type(df) == pd.core.frame.DataFrame:
        df['char_counts'] = df['text'].apply(lambda x: charcounts(x))
        df['word_counts'] = df['text'].apply(lambda x: wordcounts(x))
        df['avg_wordlength'] = df['text'].apply(lambda x: avg_wordlength(x))
    return df

def word_frequency(df, col):
	text = ' '.join(df[col])
	text = text.split()
	freq = pd.Series(text).value_counts()
	return freq