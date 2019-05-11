from Word2VecVectorizer import Word2VecSimple


class Word2VecVectorizerB:
    def __init__(self, min_count=1, size=50, window=4, vectorization_function="maxmin",sg=1):      
        self.vectorizer = Word2VecSimple(min_count=min_count, size=size, window=window, vectorization_function="maxmin",sg=sg)
        self.vectorizer.vocabulary_ = self.vectorizer.vocabs

    def fit(self, sentences):       
        new_sentences = [' '.join(x) for x in sentences]       
        self.vectorizer.fit(new_sentences)
        
    def transform(self, sentences):
        new_sentences = [' '.join(x) for x in sentences]
        return self.vectorizer.transform(new_sentences)

