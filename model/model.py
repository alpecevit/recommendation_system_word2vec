import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from gensim.models import word2vec

from functions.callback import loss_calculator
from functions.data_grouping import data_grouping_function
import pickle


min_count = 1 # including all product views to the corpus
window = 10 # a window of 10 because I will recommend 10 products per product
workers = 1 # for reproducibility 
sg = 1 # using skip-gram model because it works better with predicting multiple outputs from a single input
hs = 0 # using negative sampling because I am making the assumption that every item user interacts is positive
       # and those items that the user does not interact can be considered negative

ns_exponent = -0.50 # choosing a negative value to negative sample unpopular items
cbow_mean = 0 # I am using negative sampling and not cbow thus, cbow_mean is 0
shrink_windows = False # shrink_windows chooses a random window size between 1 and the provided window input (10)
                       # however, since we are not dealing with actual text data, product_ids in a particular sentence
                       # that are close to each other more than others does not mean they are contextually similar
                       # changing default option ensures that our provided window size will be used for every product

seed = 100 # or reproducibility 

epochs = 30 # since there is no built-in early stopping in word2vec, I visualize loss per epoch to determine the number
            # of epochs
negative = 15
sample = 0 # I do not want downsampling of popular items because popularity of an item might indicate its purchase
           # propensity and recommending it would be useful
           # also dataset consists mostly of products that are viewed only once, I am trying to balance the sample by
           # using hyperparameters (negative sampling etc.)

compute_loss=True # for computing loss per epoch
callbacks = loss_calculator() # defining a loss calculator to print out training loss after every epoch
                              # however, loss computing is buggy and there is still an open issue about it
                              # I am only using loss function to see the decreasing loss in my model

# Open issues about gensim Word2Vec training loss calculation:
# https://github.com/RaRe-Technologies/gensim/pull/2135
# https://github.com/RaRe-Technologies/gensim/issues/2617

# train word2vec model
wv_model = word2vec.Word2Vec(min_count=min_count, window=window, workers=workers, sg=sg, hs=hs,negative=negative,
                             ns_exponent=ns_exponent, cbow_mean=cbow_mean, shrink_windows=shrink_windows,
                             seed=seed, epochs=epochs, compute_loss=compute_loss, sample=sample)


if __name__ == '__main__':
    data_1 = pd.read_csv('data/final_data_1.csv')
    data_2 = pd.read_csv('data/final_data_2.csv')
    data_3 = pd.read_csv('data/final_data_3.csv')
    data = pd.concat([data_1, data_2, data_3])
    data['visitorid'] = data['visitorid'].astype(str)
    data['itemid'] = data['itemid'].astype(str)
  

    word2vec_inputs = data_grouping_function(data, 'visitorid', 'itemid')['products'].tolist()

    wv_model.build_vocab(word2vec_inputs, progress_per=10000)

    wv_model.train(word2vec_inputs, total_examples=wv_model.corpus_count, epochs=wv_model.epochs,
                   compute_loss=compute_loss, callbacks=[callbacks])

    history = pd.DataFrame(callbacks.losses, columns=['loss']).reset_index()
    history.plot(x='index', y='loss')
    plt.ylabel('training loss')
    plt.xlabel('epoch')
    plt.show()

    with open('service/model.pickle', 'wb') as f:
        pickle.dump(wv_model, f)
