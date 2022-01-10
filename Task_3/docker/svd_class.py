import pandas as pd
import numpy as np
import pickle
import math

from scipy.sparse import csc_matrix, csr_matrix
from sparsesvd import sparsesvd

USER_COL = 'user_id'
ITEM_COL = 'product_id'
DEFAULT_RATING_COL = 'rating'
DEFAULT_K = 10
DEFAULT_M = 90

class SVDRecommender:

    def __init__(self):
        pass

    def fit(self, train_df, 
            col_user=USER_COL, 
            col_item=ITEM_COL, 
            col_rating=DEFAULT_RATING_COL,
            m=DEFAULT_M) -> None:
        """
        perform train procedure for the recommendation algorithm
        :param col_item: name for items column
        :param col_user: name for user column
        :param col_rating: name for rating column
        :param train_df: pandas data frame with users, items and ratings columns
        :param m: number of most significant features
        :return: None
        """
        self.train_df = train_df.copy()
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        
        self.items = np.unique(self.train_df[self.col_item].values)
        item_to_encodeditem = {x: i for i, x in enumerate(self.items)}
        self.train_df[self.col_item] = self.train_df[self.col_item].map(item_to_encodeditem)
        
        self.users = np.unique(self.train_df[self.col_user].values)
        self.user_to_encodeduser = {x: i for i, x in enumerate(self.users)}
        self.train_df[self.col_user] = self.train_df[self.col_user].map(self.user_to_encodeduser)

        sparse_matrix = csc_matrix((self.train_df[self.col_rating], 
                            (self.train_df[self.col_user].values, self.train_df[self.col_item].values)), 
                            shape = (len(self.users), len(self.items)))
        
        U, S, Vt = sparsesvd(sparse_matrix, m)
        
        S_diag = np.zeros((len(S), len(S)), dtype=np.float32)
        
        for i in range(len(S)):
            S_diag[i, i] = math.sqrt(S[i])
    
        self.U = csr_matrix(np.transpose(U), dtype=np.float32)
        
        S_diag = csr_matrix(S_diag, dtype=np.float32)
        Vt = csr_matrix(Vt, dtype=np.float32)
        self.right_term = S_diag * Vt

    def predict(self, test_df, col_user=USER_COL, k=DEFAULT_K) -> pd.DataFrame:
        """
        predicts recommendations for test users
        :param test_df: pandas data frame with users column
        :param k: number of items to predict per user
        :return: prediction pandas data frame with two columns first contains user's
        and second contains list of recommended items
        """        
        test_df_new = test_df.copy()
        estimatedRatings = []
        
        encodeditem_to_item = {i: x for i, x in enumerate(self.items)}
        
        test_df_new[col_user] = test_df_new[col_user].map(self.user_to_encodeduser)
        user_list = test_df_new[col_user].values
        
        for i in user_list:
            prod = self.U[i, :] * self.right_term
            prod_dense = prod.todense()
            prod_dense_top = (-prod_dense).argsort()[0, :k].tolist()[0]
            prod_dense_top_encoded = [encodeditem_to_item.get(prod_dense_top[j]) for j in range(k)]
            estimatedRatings.append(prod_dense_top_encoded)
        
        encodeduser_to_user = {i: x for i, x in enumerate(self.users)}
        test_df_new[col_user] = test_df_new[col_user].map(encodeduser_to_user)
        
        test_df_new['svd'] = estimatedRatings
        return test_df_new

if __name__ == '__main__':
    pivot_table = pd.read_pickle('data/pivot_table.pkl')
    df_test = pd.read_pickle('data/df_test.pkl')

    svd = SVDRecommender()
    svd.fit(df_test)
    pickle.dump(svd, open("models/svd.pickle.dat", "wb"))