import pandas as pd 
import numpy as np 

class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.item_id = None
        self.interaction = None
        self.popularity_recommendations = None
    def create(self, train_data, item_id, interaction):
        self.train_data  = train_data
        self.item_id = item_id
        self.interaction = interaction

        item_ids = pd.unique(self.train_data[self.item_id])
        df = self.train_data.groupby(self.item_id).agg({interaction: 'count'}).reset_index()
        df.sort_values(by='feedBack', ascending=False, inplace=True)
        self.popularity_recommendations = df

        # self.popularity_recommendations = df.sort_values(by=interaction, ascending=False, inplace=True)
        # return df
    def recommend(self, top):
        # print('pop', self.popularity_recommendations)
        return self.popularity_recommendations[:int(top)][self.item_id].to_list()