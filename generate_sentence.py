
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict


# In[2]:


train_datasets = ['gutenberg']
test_datasets  = ['gutenberg']
prefix = '_'.join(train_datasets) + '-' + '_'.join(test_datasets)

word_to_id = np.load('%s_word_to_id.npy' % prefix).item()
id_to_word = np.load('%s_id_to_word.npy' % prefix).item()
n_grams = np.load('%s_n_grams.npy' % prefix).item()
train_sentences = np.load('%s_train.npy' % prefix)
valid_sentences = np.load('%s_valid.npy' % prefix)


# In[3]:


# Store conditional counts
conditional_n_grams = dict()

for n in range(1,2+1):
    conditional_n_grams[n] = defaultdict(dict)
    
    for n_gram, count in n_grams[n].iteritems():
        conditional_n_grams[n][n_gram[:-1]][n_gram[-1]] = count


# In[4]:


# Store continuation counts
continuations = dict()

for n in range(1,2+1):
    continuations[n] = defaultdict(set)
    
    for n_gram, count in n_grams[n].iteritems():        
        if count > 0:
            continuations[n][n_gram[1:]].add(n_gram[0])


# In[5]:


continuation_prob = np.zeros((len(word_to_id) + 1,))
denominator = sum([len(x) for x in continuations[n].values()])

for i in range(len(word_to_id) + 1):
    continuation_prob[i] = float(len(continuations[2][(i,)])) / denominator


# In[6]:


def get_distribution(context, d=0.75):
    n = len(context) + 1
    assert n == 2
    
    # create probability distribution by copying continuation probability
    counts = np.array(continuation_prob)
    
    # premultiply with lambda_
    lambda_ = float(d)/n_grams[n-1][context] * len(conditional_n_grams[n][context])
    counts *= lambda_
    
    for i in conditional_n_grams[n][context]:
        bigram_score = float(max(conditional_n_grams[n][context][i] - d, 0))/n_grams[n-1][context]
        counts[i] += bigram_score
        
    return counts


# In[12]:


def sample_from_distrib(context):
    n = len(context) + 1
    
    # normalize to construct distribution
    probabilities = get_distribution(context)
    #print context, sum(probabilities)
    choice = np.random.choice(len(word_to_id) + 1, p=probabilities)
    
    return choice


# In[115]:


tokens = [word_to_id['@@start@@'],]

while True:
    try:
        while len(tokens) < 10:
            try:
                tokens.append(sample_from_distrib((tokens[-2],tokens[-1],)))
            except:
                try:
                    tokens.append(sample_from_distrib((tokens[-1],)))
                except KeyError:
                    tokens.append(sample_from_distrib(tuple()))

            if id_to_word[tokens[-1]] == '@@end@@':
                break
    except KeyError:
        continue
    else:
        break


# In[116]:


for id_ in tokens:
    print id_to_word[id_],

