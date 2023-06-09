import nltk
import ir_datasets
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize


lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
tfidf_vect = TfidfVectorizer()
index=[]
tfidfV=[]
inverted_index = defaultdict(list)

def lemmatization(tokens):
    term = [lemmatizer.lemmatize(word) for word in tokens]
    return term
    
def tfidf(dataset):
    # TFIDF
    x = tfidf_vect.fit_transform(dataset)
    print(x.shape)
    return x

def inverted_indexx(dataset):
    # inverted_index = defaultdict(list)
    for i,doc in enumerate (dataset):
        for term in doc:
            inverted_index[term].append(i)
    search_term ="sadam"
    if search_term in inverted_index:
        matching_documents = inverted_index[search_term]
        print(f"Documents containing '{search_term}' : {matching_documents}")
    else:
        print(f"No Documents contain '{search_term}' ")
    return inverted_index

def preprocess(dataset_name):
    print(f"hi {dataset_name}")
    dataset = pd.read_csv( fr"C:\Users\pc\.ir_datasets\{dataset_name}\collection.tsv" , encoding='utf-8',sep='\t', header=None ) 
    dataset.columns = ('key', 'document' )
    dataset['cleaned_document'] = dataset['document'].apply(lambda x: [word.lower() for word in str(x).split() if word.lower() not in stop_words])
    print(dataset.head())
    # dataset['lemmatized_document'] = dataset['cleaned_document'].apply(lambda x:lemmatization(x))
    dataset['lowered_document'] = dataset['cleaned_document'].apply(lambda x: " ".join(x))
    print(dataset.head())
    # # TFIDF
    tfidf(dataset['lowered_document'])
    #inverted_index
    inverted_indexx(dataset['cleaned_document'])
    return(tfidf(dataset['lowered_document']))

# preprocess("antique")

# def query_Processing(query):
#      # Tokenize the document
#     tokens = word_tokenize(query.lower())
#     # Remove stop words and punctuation
#     tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
#     # Lemmatize the tokens
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     # Join the tokens back into a string
#     query_terms = defaultdict(int)
#     for term in tokens:
#         query_terms[term] += 1
#     query_terms_list = list(query_terms.keys())
#     #Retrive the documents that contain the query terms
#     matching_docs =[]
#     for term in query_terms_list:
#         if term in index:
#             matching_docs.extend(index[term])
#     #remove doublicate from the matching_docs list
#     matching_docs = list(set(matching_docs))
#     #coumpute the similarity between the query and the matching documents
#     similarety_scors = []
#     for doc_idx in matching_docs :
#         doc_terms =index[doc_idx]
#        # print ("term :" ,doc_terms)
#        preprocess("antique")
#         score = 0
#         for term in query_terms_list:
#             if term in  tfidf_vect.vocabulary_ :
#                 term_idx =  tfidf_vect.vocabulary_[term]
#                 score += tfidfV[doc_idx , term_idx] * query_terms[term]
#         similarety_scors.append((doc_idx , score))
#     #sort matching documents by similarety scors
#     similarety_scors.sort(reverse = True , key = lambda x: x[1])
#     key = 6
#     for i in range(key):
#         doc_idx = similarety_scors[i][0]
#         print ("Documents " ,doc_idx ," with the similarety score " , similarety_scors[i][1],"\n DOCUMENT: ", dataset['document'][doc_idx])

# query_Processing('how can we get concentration onsomething?')

def query_preprocess(query):
    porter= PorterStemmer()
    tokens =query.split()
    stemmed_tokens =[porter.stem(token.lower()) for token in tokens if token.lower() not in stop_words]
    print(stemmed_tokens)
    return ' '.join(stemmed_tokens)

# query_preprocess('how can we get concentration onsomething?')

# Query Preprocess
def handle_antique_query():
    dataset = ir_datasets.load("antique/test")
    queries =[]
    queries_id =[]
    for query in dataset.queries_iter():
        queries_id.append(query[0])
        queries.append(query[1])
    #query Tfidf
    query_vector = tfidf_vect.transform([query_preprocess(queries[0])])
    print(query_vector.shape)
    return(query_vector)

def handle_ms_query():
    dataset = ir_datasets.load("msmarco-passage/queries.train")
    queries =[]
    queries_id =[]
    for query in dataset.queries_iter():
        queries_id.append(query[0])
        queries.append(query[1])
    #query Tfidf
    query_vector = tfidf_vect.transform([query_preprocess(queries[0])])
    print(query_vector.shape)
    return(query_vector)

# handle_antique_query()

# Cosine Similarity
def calculate_cos_similarity(dataset_name):
    cosine_similarities =[]
    tfidf_matrices = preprocess(dataset_name)
    tfidf_query = handle_antique_query()
    for tfidf_matrix in  tfidf_matrices:
        cosine = cosine_similarity(tfidf_query,tfidf_matrix).flatten()
        cosine_similarities.append(cosine)
    #Most Similar (rank)
    most_similar_documents = sorted(range(len(cosine_similarities)), key= lambda i : cosine_similarities[i],reverse=True)
    print(most_similar_documents)

# calculate_cos_similarity("antique")

retrived_docs=[]
# def my_retrived():
#     for q in queries:
#         retrived_docs.append(query_Processing(q))
#     print(retrived_docs)
#     retrived=len(retrived_docs)
#     print(retrived)

# def precision_rank(queries,retrieved_docs):
#     num_queries = len(queries)
#     num_relevant_docs_at_10 = 0
#     for query, docs in zip(queries, retrieved_docs):
#         relevant_docs = set(relevant_docs_for_query[query])
#         retrieved_docs_at_10 = set(docs[:10])
#         num_relevant_docs_at_10 += len(relevant_docs.intersection(retrieved_docs_at_10))

#     precision_at_10_score = precision(num_relevant_docs_at_10, 10 * num_queries)

# # Define the MRR metric
# def mrr(y_true, y_pred):
#     ranks = [i+1 for i, doc_id in enumerate(y_pred) if doc_id in y_true]
#     if len(ranks) == 0:
#         return 0
#     else:
#         return 1.0 / ranks[0]

# # Evaluate the IR model using MRR
# queries2 = [tfidf_vect.transform(queries)]
# labels = [[1], [2], [3]]
# mrr_scorer = make_scorer(mrr, greater_is_better=True)
# mrr_scores = cross_val_score(rank_documents, queries2, labels, cv=3, scoring=mrr_scorer)
# print('MRR scores:', mrr_scores)
# print('Mean MRR:', mrr_scores.mean())

def serach_result(query):
    if query in inverted_index:
        matching_documents = inverted_index[query]
        print(f"Documents containing '{query}' : {matching_documents}")
    else:
        print(f"No Documents contain '{query}' ")
    return matching_documents
# serach_result("sadam")

def handle_qrels2():
    qrels_df2=pd.read_csv( fr"C:\Users\pc\.ir_datasets\msmarco-passage\qrels.train",encoding='utf-8', sep=' ',header =None)
    qrels_df=pd.read_csv( fr"C:\Users\pc\.ir_datasets\antique\test\qrels.TSV",encoding='utf-8', sep=' ',header =None)
    qrels_df.columns = ('query_id' , 'iteration' , 'doc_id' ,'relevance')
    print(qrels_df.head())
    print(len(qrels_df))

# handle_qrels()

