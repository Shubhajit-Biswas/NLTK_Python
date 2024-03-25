import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#nltk.download('punkt')

sentences = ['I bought a pear',
             'I eat a banana',
             'I ate a banana',
             'I ate a pear',
             'I carry a banana',
             'He ate a pear',
             'He bought a banana',
             'She eats a pear',
             'They ate a banana',
             'We bought a pear',
             'She ate a banana',
             'He carries a banana',
             'They ate a pear',
             'She ate a orange',
             'We bought a banana',
             'He carries a orange',
             'They carry a orange',
             'They carry a banana',
             'They carry a pear']
bag_words = ' '.join(sentences).lower()
vocab = list(set(bag_words.split(' ')))
vocab.sort()
#print(vocab)
vocab_size = len(vocab)
word_co_occurances = np.zeros([vocab_size,vocab_size])
#word_co_occurances_df = pd.DataFrame(word_co_occurances, index = vocab, columns = vocab)
#print (word_co_occurances_df)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
n_neighbour = 1

for i,word in enumerate(vocab):
    for j,pair_word in enumerate(vocab):
        for sent in tokenized_sentences:
            for word_index,word_sent in enumerate(sent):
                if word==word_sent:
                    first_position = max(word_index-n_neighbour, 0)
                    max_position = min(word_index-n_neighbour+1,len(sent))
                    
                    for neighbour in sent[first_position:max_position]:
                        if neighbour==pair_word:
                            word_co_occurances[i,j]+=1
for i in range(0,vocab_size):
    word_co_occurances[i,i] = 0
    
word_co_occurances_df = pd.DataFrame(word_co_occurances, index = vocab, columns = vocab)
#print (word_co_occurances_df)

cosine_similarity_df = pd.DataFrame(cosine_similarity(word_co_occurances_df), index = vocab, columns = vocab)
print(cosine_similarity_df)
