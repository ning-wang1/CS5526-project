from TfidfVectorizerB import TfidfVectorizerB
from Word2VecVectorizerB import Word2VecVectorizerB
import  nltk, os, re
import json
import pandas as pd
import numpy as np
import pickle
import unicodedata
from matplotlib import pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

nltk.download('stopwords')
nltk.download('wordnet')

class Vectorizer:
    
    def __init__(self, name, vectorizer_type):
        
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.name = name 
        
        tfidf = TfidfVectorizerB(stop_words="english", min_df=0.003, max_df=0.997)
        # w2v = W2V
        w2v_skipgram = Word2VecVectorizerB(min_count=1, size=50, window=4, vectorization_function="maxmin",sg=1)
        w2v_cbow = Word2VecVectorizerB(min_count=1, size=50, window=4, vectorization_function="maxmin",sg=0)
        self.vectorizers = {
            'tfidf': tfidf,
            'w2v_skipgram': w2v_skipgram,
            'w2v_cbow': w2v_cbow
        }
        
        self.vectorizer = self.vectorizers[vectorizer_type]

    def vectorize(self, input_file):
        input_file_f = open(input_file)
        data = json.load(input_file_f)
        input_file_f.close()
        ## Preprocess the data into an array which can be processed by w2v
        tokenSet = []
        magLabels = []
        eqIDs = []
        tweetIDs = []
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # Keep only alphanumeric characters as tokens
        for idx_e in range(len(data)):
            for idx_n in range(len(data[idx_e]["tweets"])):
                text = data[idx_e]["tweets"][idx_n]["text"] 
                text = strip_accents(text)        
                text = re.sub(r"http\S+", "", text)
                '''
                for i, thing in enumerate(text):
                    try:
                        if "/" in thing:
                            text.remove(thing)
                    except UnicodeEncodeError, e:
                        print("Unicode error", e)
                        text.remove(thing)
                        '''
                text = text.lower()  # convert to lowercase
                tokens = tokenizer.tokenize(text)  # tokenize
                tokenSet.append(tokens)
                magLabels.append(data[idx_e]["magnitude"])  # every tokenized tweet has a magnitude label
                eqIDs.append(data[idx_e]['id'])
                tweetIDs.append(data[idx_e]['tweets'][idx_n]['id'])

        # Remove stopwords, numbers, singleton characters, and lemmatize
        stopwords_nltk = nltk.corpus.stopwords.words('english')
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        tokenSet = [[lemmatizer.lemmatize(token) for token in doc if
                     not (token in stopwords_nltk or token.isnumeric() or len(token) <= 1)] for doc in
                    tokenSet]  # remove stopwords
        print(type(tokenSet))

         ############ word cloud begin
        print(tokenSet[1])
        a='aaaa'
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(a) 
  
        # plot the WordCloud image                        
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
          
        plt.show() 

        ############# word cloud end


        print('Preprocessing Completed. Total earthquakes: ', len(data), '. Total tweets: ', len(tokenSet))
        
        nSamples = len(tokenSet)
        #print(tokenSet)
        #subsample = 20
        
        #Here we fit the vectorizer
        self.vectorizer.fit(tokenSet[0:nSamples])
        
        #Now we focus on adding the label:
        sentences = self.vectorizer.transform(tokenSet[0:nSamples])

        print('Vocabulary: ')
        #print(self.vectorizer.vectorizer.vocabulary_)
        
        sentences_array = sentences.toarray()
        print(sentences_array.size,' ',sentences_array.shape)


        labels_array = np.array(magLabels[0:nSamples])
        labels_array = np.expand_dims(labels_array,axis=1)
        eqID_array = np.array(eqIDs[0:nSamples])
        eqID_array = np.expand_dims(eqID_array,axis=1)
        tweetIDs_array = np.array(tweetIDs[0:nSamples])
        tweetIDs_array = np.expand_dims(tweetIDs_array, axis=1)

        # print("labels_array dims")
        #print(labels_array.shape)q
        # print("sentence_array dims")
        #print(sentences_array.shape)
        sentences_array = np.append(sentences_array,labels_array,axis=1)
        sentences_array = np.append(sentences_array,eqID_array,axis=1)
        sentences_array = np.append(sentences_array, tweetIDs_array, axis=1)

        
        
        # print(sentences_array[0:2])
        # print("finished vectorizing")
        
        df = pd.DataFrame(sentences_array)
        num_cols = df.shape[1]
        df = df.rename(columns={ df.columns[num_cols-3]: 'y' , df.columns[num_cols-2]: 'eqID' , df.columns[num_cols-1]: 'tweetID'})
        #df = df.rename(columns={df.columns[num_cols-2]: 'y'})
        #df = df.rename(columns={ df.columns[num_cols-1]: 'eqID' })
        self.model_df = df
        #print(df)
        
        
    def save_model(self):
        # Either use the specified models name or pick a
        # numbered models name that has not been used yet
        model_dir = self.dir+'/models/vecs'
        print('Saving models ...')

        object_to_be_saved = self.model_df
        rows_count = len(self.model_df.index)
        

        
        if self.name:
            filename = model_dir + '/' + self.name + '.pickle'
        else:
            files = os.listdir(model_dir)
            already_used = True
            i = 0
            while already_used:
                filename = 'model_'+ str(i) + '.pickle'
                if filename in files:
                    i += 1
                else:
                    already_used = False
            filename = model_dir + '/' + filename
        with open(filename, 'wb') as f:

            vectorized_model = (self.model_df, self.vectorizer.vectorizer.vocabulary_)
            pickle.dump(vectorized_model, f, protocol=2)
            print("Model saved " + filename)
