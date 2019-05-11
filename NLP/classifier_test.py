from Classifier import Classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm



cls = RandomForestRegressor()
cls_SVR = SVR(gamma='auto')
cls_MLPR_1 = MLPRegressor(hidden_layer_sizes=(250, 150, 20))
cls_MLPR_2 = MLPRegressor(hidden_layer_sizes=(250, 20))
cls_MLPR_3 = MLPRegressor(hidden_layer_sizes=(500, 30))
cls_MLPR_4 = MLPRegressor(hidden_layer_sizes=(500, 300, 20))
classifier = Classifier()
models=[

    #{
        #'name':"mlp1_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",

        #'name':"mlp1_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018",

        #'name':"mlp1_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",
	
        #'name':"mlp_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",

        # 'name':"mlp_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",
        
        #'classifier': cls_MLPR_1
    #},
  
    {

        #'name':"mlp2_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",

        'name':"mlp2_w2v_world_2018",
        'vectorizer_pickle_filename': "w2v_world_2018_cbow",

        #'name':"mlp2_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",

        #'name':"mlp_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",

        # 'name':"mlp_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",
        
        'classifier': cls_MLPR_2
    },
    #{
        
        #'name':"mlp3_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",

        #'name':"mlp3_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018",

        #'name':"mlp3_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",

    	
        #'name':"mlp_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",

        # 'name':"mlp_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",
        
        #'classifier': cls_MLPR_3
    #},
    
    #{
        #'name':"mlp4_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_cbow",

        #'name':"mlp4_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",

        #'name':"mlp4_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",
    	
        #'name':"mlp4_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",

        #'name':"mlp4_w2v_world_2017",
        #'vectorizer_pickle_filename': "w2v_world_2017_cbow",

        # 'name':"mlp_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",
        
        #'classifier': cls_MLPR_4
    #},
    {
        #'name':"svr_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_cbow",

        #'name':"svr_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",

        #'name':"svr_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",

        #'name':"svr_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",
        'name':"svr_w2v_world_2017",
        'vectorizer_pickle_filename': "w2v_world_2017_cbow",

        # 'name':"svr_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",

        'classifier': cls_SVR
    },
    
    {
        #'name':"randomforest_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_cbow",

        #'name':"randomforest_w2v_world_2018",
        #'vectorizer_pickle_filename': "w2v_world_2018_skipgram",

        #'name':"randomforest_w2v_world_2019",
        #'vectorizer_pickle_filename': "w2v_world_2019",

        #'name':"randomforest_tfidf_world_2018",
        #'vectorizer_pickle_filename': "tfidf_world_2018",

        'name':"randomforest_w2v_world_2017",
        'vectorizer_pickle_filename': "w2v_world_2017_cbow",

        # 'name':"randomforest_tfidf_world_2009-2019",
        # 'vectorizer_pickle_filename': "tfidf_world_2009-2019",

        'classifier': cls
    }
]
classifier.evaluate_models(models)


def plot_result():
    plt.figure(1)
    x = np.linspace(0,1.8)
    names = ('mlp_tfidf','svr_tfidf','random_forest_tfidf','mlp_cbow','svr_cbow','random_forest_cbow')
    colors = ('green','red','blue')
    files = ('mlp','svr','randomforest')
    str1 = '_tfidf_world_2018.txt'
    str2 = '_w2v_world_2018.txt'
    i = 0
    for i in range(3):
        err = np.loadtxt('prediction_error/'+files[i]+str1)
        ecdf = sm.distributions.ECDF(err)
        y = ecdf(x)
        plt.plot(x,y,color=colors[i], linestyle = '-', label=names[i])
        i+=1
    i = 0
    for i in range(3):
        err = np.loadtxt('prediction_error/'+files[i]+str2)
        ecdf = sm.distributions.ECDF(err)
        y = ecdf(x)
        plt.plot(x,y,color=colors[i], linestyle = ':', label=names[i+3])
        i+=1

    plt.legend()
    plt.xlabel('prediction erro')
    plt.ylabel('CDF')
    plt.show()

#plot_result()


