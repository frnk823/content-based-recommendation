import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json


def train(data_source):
    start = time.time()
    ds = pd.read_csv(data_source,encoding='utf-8', sep="|")
    print("Training data ingested in %s seconds." % (time.time() - start))

    start = time.time()
    _train(ds)
    print("Engine trained in %s seconds." % (time.time() - start))

def _train(ds):
    """
    Train the engine.

    Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each product. The 'stop_words' param
    tells the TF-IDF module to ignore common english words like 'the', etc.

    Then we compute similarity between all products using SciKit Leanr's linear_kernel (which in this case is
    equivalent to cosine similarity).

    Iterate through each item's similar items and store the 100 most-similar. Stops at 100 because well...
    how many similar products do you really need to show?

    Similarities and their scores are stored in redis as a Sorted Set, with one set for each item.

    :param ds: A pandas dataset containing two fields: description & id
    :return: Nothin!
    """
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['content'])

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    ds['sim_artcle_list']=''
    for idx in range(len(ds)):
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [{"sim":cosine_similarities[idx][i], "id":ds['id'][i]} for i in similar_indices]
        # First item is the item itself, so remove it.
        # This 'sum' is turns a list of tuples into a single tuple: [(1,2), (3,4)] -> (1,2,3,4)

        ds.loc[idx,'sim_artcle_list']= json.dumps(similar_items[1:])
    # print(ds)
    ds.to_csv('sim_list.txt',sep='|',index=False)


train('corpus.txt')

# ds = pd.read_csv('sim_list.txt',encoding='utf-8', sep="|")
# a= ds.iloc[2]['sim_artcle_list']
# print(type(a),a)
# b=json.loads(a)
# print(type(b),b)
# # print(json.loads(a))
