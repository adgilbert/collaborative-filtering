import pandas as pd
import numpy as np
from scipy import cluster

def link_for_metro(user_df, business_df, review_df):

# each row is a user review of a restaurant, columns are:
# 1) user_id 2) business_id 3) city (of business) 4) (user) review_count

    linked_review = pd.merge(review_df[['user_id','business_id']],
                             business_df[['business_id','city',
                                          'latitude', 'longitude']],
                             on='business_id')
    return pd.merge(linked_review,
                    user_df['user_id', 'review_count'],
                    on='user_id')


def add_metro_area(df, K=12):
    _, df['metro_area'] = cluster.vq.kmeans2(df, K)
    return df

def calc_perc_metro(df, K=12):
    counted = df.groupby(['user_id','metro_area']).count()
    for i in range(0,K):
        df[('city_perc_'+str(i))] = counted['user_id'].loc[
                                                    counted['metro_area'==i]]



#aggregate by user_id

# use K = 12 (number of metro areas)


#for each user_id, compute percentage for each city
