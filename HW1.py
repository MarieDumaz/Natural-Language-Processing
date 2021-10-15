# CS 593A: Natural Language Processing
# Assignment 1

from nltk.tokenize import TweetTokenizer
from nltk import bigrams
from nltk import ngrams
from collections import Counter

#opens file
f = open("movie_lines.txt", "r")
text = f.read()

#split text into lines
text = text.split("\n")

#remove last entry if null
if text[-1] == '':
    text = text[:len(text)-2]

lines = []
#read file line by line
for line in text:
    #split on the pattern
    conv = line.split("+++$+++")[4]
    #remove whitespaces in front or rear of string
    conv = conv.strip()
    #convert to lowercase
    conv = conv.lower()
    #tokenize
    conv = TweetTokenizer().tokenize(conv)

    #add all words to one list
    for x in conv:
        lines.append(x)

#counts unigrams    
unigrm_count = Counter(lines)
#sort
unigrm = unigrm_count.most_common()
print("TOP 10 UNIGRAMS: \n", unigrm[:10])

#counts bigrams
bigrm_count = Counter(list(bigrams(lines)))
#sort
bigrm = bigrm_count.most_common()
print("TOP 10 BIGRAMS: \n", bigrm[:10])

#counts trigrams
trigrm_count = Counter(list(ngrams(lines, 3)))
#sort
trigrm = trigrm_count.most_common()
print("TOP 10 TRIGRAMS: \n", trigrm[:10])
