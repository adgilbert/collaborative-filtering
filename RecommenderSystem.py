
# coding: utf-8

# In[136]:

from __future__ import division
import pandas as pd
import numpy as np
import os
from scipy import cluster
import pickle
# import plotly.plotly as py
# import matplotlib.pyplot as plt
# %matplotlib inline
import math
from sklearn import preprocessing
import get_metro_features as gmf
import get_city_proportion as gcp
import plot_and_split as pas
import datetime
import scipy
from scipy.sparse import csc_matrix


# In[155]:
cur_dir = os.getcwd()
dataset_dir = cur_dir
ext = ''
# cur_dir = '~/Desktop/dataset'
# dataset_dir = cur_dir + '/head'
# ext = '_head'
CITYNAME = 'LasVegas'
skip = True

if skip:
    variables = pickle.load(open('data/variables.pck', 'rb'))
    cities=variables['cities']
    connections=variables['connections']
    user_res=variables['user_res']
    major_businesses=variables['major_businesses']
    CITYNAME=variables['CITYNAME']
else:



    # In[156]:

    users = pd.read_csv(dataset_dir + '/user' + ext + '.csv')
    reviews = pd.read_csv(dataset_dir + '/review' + ext + '.csv')
    businesses = pd.read_csv(dataset_dir + '/business' + ext + '.csv')


    # In[157]:

    major_businesses, state_num = pas.eliminate_minor_states(businesses)
    # Manually define boundaries because that's the easiest way to do this
    boundaries = dict(AZ=dict(bot=31.30, top=37.0, left=-115.0, right=-109),
                     NC=dict(bot=33.8, top=36.8, left=-84.4, right=-75.1),
                     PA=dict(bot=39, top=42.3, left=-80.7, right=-74.6),
                     NV=dict(bot=34.9, top=42.07, left=-120.1, right=-114.02)) 
    # One odd point this doesn't clean for NV because of diagonal, but whatever, not worth a more advanced function
    for state in state_num:
        if state not in boundaries.keys():
            # All other states are clean
            continue
        # Filter the one's that do need to be filtered
        major_businesses = major_businesses[(major_businesses.state != state) | (major_businesses['latitude'] > boundaries[state]['bot'])]
        major_businesses = major_businesses[(major_businesses.state != state) | (major_businesses['latitude'] < boundaries[state]['top'])]
        major_businesses = major_businesses[(major_businesses.state != state) | (major_businesses['longitude'] > boundaries[state]['left'])]
        major_businesses = major_businesses[(major_businesses.state != state) | (major_businesses['longitude'] < boundaries[state]['right'])]

    # Also NC and SC should be combined, because its the same area 
    major_businesses.loc[major_businesses.state=='SC', 'state'] = 'NC'


    # In[158]:

    states = set(major_businesses['state'])
    print(states)
    init = np.zeros((len(states), 2))
    for i, state in enumerate(states):
        init_pt = major_businesses[major_businesses['state'] == state].sample(1)
        init[i, 0] = init_pt['latitude']
        init[i, 1] = init_pt['longitude']

    # It looks like we have one NaN in lat/lng
    major_businesses.dropna(subset=['latitude', 'longitude'], inplace=True)
    clusters = pas.cluster_cities(major_businesses, k=11, iter=500, init=init)


    # In[159]:

    major_businesses = major_businesses.assign(metro_area=pd.Series(clusters[1]).values)
    major_businesses = major_businesses[major_businesses['review_count'] > 30] #impose 30 review minimum for business inclusion

    # In[160]:

    # b = major_businesses.keys()
    # nonA = [ba for ba in b if 'Attributes' not in ba and 'attributes' not in ba and 'hours' not in ba]
    # major_businesses2 = major_businesses[nonA]
    # # major_businesses2.groupby('metro_area')
    # major_businesses2 = major_businesses2.rename(columns={'review_count':'biz_review_count'})


    # In[161]:

    join_date = pd.to_datetime(users['yelping_since']).dt.date
    now_date = datetime.date(2017, 12, 1)
    users['weeks_on_yelp'] = (now_date - join_date).dt.days / 7
    print(users.shape)
    # Eliminate users with less than 20 reviews
    users = users[users.review_count >= 20]
    print(users.shape)


    # In[162]:

    connections = gmf.link_for_metro(users, major_businesses, reviews)
    # This gives us percentage of reviews in each metro area
    perc_metro = gmf.calc_perc_metro(connections)
    # print(perc_metro)




    # In[163]:

    num_visited = gmf.num_metros_visited(perc_metro)
    weeks_metro = gmf.calc_num_weeks_metro(num_visited)


    # In[164]:

    # review_metro = gmf.reviews_per_week_per_metro(weeks_metro)
    user_features = gmf.define_user_features(users, weeks_metro, clusters[0].shape[0])
    # user_features['num_metros_visited'] = perc_metro.groupby('user_id')['num_metros_visited'].nunique()
    # user_features = pd.merge(user_features, dnew, on='user_id', how='left')
    user_features.fillna(0, inplace=True)
    user_features = user_features.groupby('user_id').max()
    user_features = user_features.reset_index()


    # In[165]:

    user_f = np.array(user_features[['m0_percent', # removed 'num_metros_visited', 'weeks_on_yelp', 
           'm0_weeks', 'm1_percent', 'm1_weeks', 'm2_percent', 'm2_weeks',
           'm3_percent', 'm3_weeks', 'm4_percent', 'm4_weeks', 'm5_percent',
           'm5_weeks', 'm6_percent', 'm6_weeks', 'm7_percent', 'm7_weeks',
           'm8_percent', 'm8_weeks', 'm9_percent', 'm9_weeks', 'm10_percent',
           'm10_weeks']])
    user_f = user_f.astype(float)
    initialization = np.zeros((11, user_f.shape[1]))
    for i in range(11):
        initialization[i, 2*i] = 1
    print(np.max(initialization, axis=1))
    # print(initialization)
    # print(np.sum(user_f, axis=1)[0:20])
    # # print(user_features[0:5])
    # # user_fnorm = preprocessing.normalize(user_f, norm='l1', axis=0, copy=True, return_norm=False)
    user_clustering = cluster.vq.kmeans2(user_f, initialization, iter=600, minit='matrix')
    user_res = user_features.copy()
    user_res['group'] = user_clustering[1]
    print(np.max(user_clustering[0], axis=1))
    print(user_res.keys())


    # In[166]:

    user_res['cluster_dist'] = np.zeros(user_res.shape[0])
    cluster_keys = [k for k in user_res.keys() if k not in ['user_id', 'group', 'cluster_dist', 'num_metros_visited', 'weeks_on_yelp']]
    for m in range(1, clusters[0].shape[0]):
        test_group = user_res.loc[user_res.group == m, cluster_keys]
    #     print(test_group)
        user_res.loc[user_res.group == m, 'cluster_dist'] = np.linalg.norm(user_res.loc[user_res.group == m, cluster_keys] - 
                                                                    user_clustering[0][m], axis=1)

        
    # Cut out users with less than 20 reviews


    # In[167]:

    city_centers = dict(
        Toronto=(43.66, -79.58),
        LasVegas=(36.0, -115.0),
        Edinburgh=(55.0, -3.19),
        Pittsburgh=(40.12, -80.12),
        Madison=(43.07, -89.38),
        Champaign=(40.08, -88.29),
        Stuttgart=(48.71, 9.23),
        Phoenix=(33.39, -111.91),
        Cleveland=(41.18, -81.49),
        Charlotte=(35.25, -80.79),
        Montreal=(45.58, -73.54)
    )

    def closest_value(compare, lat, lng):
        mindist = 1e6
        mindex = -1
        for k in compare:
            temp = np.linalg.norm([compare[k][0] - lat, compare[k][1] - lng])
            if temp < mindist:
                mindex = k
                mindist = temp
    #             print('new mindist = {}'.format(mindist))
        return mindex


    cities = dict()
    city_bs = dict()
    for i in range(clusters[0].shape[0]):
        city_locals = user_res[user_res['group'] == i]
        connection_locals = connections[connections['user_id'].isin(city_locals['user_id'])]
        mean_lat, mean_lng = np.mean(connection_locals['latitude']), np.mean(connection_locals['longitude'])
    #     print(mean_lat, mean_lng)
        city = closest_value(city_centers, mean_lat, mean_lng)
        cities[city] = i
        print('User city {} = {}\t {:.2f}, {:.2f}'.format(city, i, mean_lat, mean_lng))
        
        # Now do businesses
        business_locals = major_businesses[major_businesses['metro_area'] == i]
        mean_lat, mean_lng = np.mean(business_locals['latitude']), np.mean(business_locals['longitude'])
        city = closest_value(city_centers, mean_lat, mean_lng)
        city_bs[city] = i
        print('Biz city  {} = {}\t {:.2f}, {:.2f}'.format(city, i, mean_lat, mean_lng))
        print('===================================')
    # print(cities)


    # In[168]:


    city_centers = """
    #: \tlat, lng =\tCity
    0:\t43.60, 79.50=\tToronto
    1:\t36.0, -115.0=\tLas Vegas
    2:\t55.0, -3.19=\tEdinburgh
    3:\t43.0, -79.9=\tPittsburgh
    4:\t43.07, -89.38=\tMadison
    5:\t40.08, -88.29=\tChampaign
    6:\t48.71, 9.23=\tStuttgart
    7:\t33.39, -111.91=\tPhoenix
    8:\t41.18, -81.49=\tCleveland
    9:\t35.25, -80.79=\tCharlotte
    10:\t45.58, -73.54=\tMontreal
    """
    # print(city_centers)


    variables = dict(
        cities=cities,
        connections=connections, 
        user_res=user_res,
        major_businesses=major_businesses,
        CITYNAME=CITYNAME
    )

    pickle.dump(variables, open('data/variables.pck', 'wb'))

def get_city_array(city_index, connections, user_res, businesses, savename=None):
    local_users = user_res[user_res['group'] == city_index]
    local_bizes = businesses[businesses['metro_area'] == city_index]
    local_connections = connections[connections['user_id'].isin(local_users['user_id'])]
    local_connections = local_connections[local_connections['business_id'].isin(local_bizes['business_id'])]
    
    # For tourist, get new users but same businesses
    tourist_users = user_res[user_res['group'] != city_index]
    tourist_bizes = businesses[businesses['metro_area'] == city_index]
    tourist_connections = connections[connections['user_id'].isin(tourist_users['user_id'])]
    tourist_connections = tourist_connections[tourist_connections['business_id'].isin(tourist_bizes['business_id'])]

    biz_ids = pd.merge(local_connections[['business_id']], tourist_connections[['business_id']], on=['business_id'], how='inner')
    local_connections = local_connections[local_connections['business_id'].isin(biz_ids['business_id'])]
    tourist_connections = tourist_connections[tourist_connections['business_id'].isin(biz_ids['business_id'])]

    # Now set up local array
    local_users = set(local_connections.user_id)
    local_businesses = set(local_connections.business_id)
    local_data = local_connections['stars'].tolist()
    local_col = local_connections.user_id.astype('category', categories=local_users).cat.codes
    local_row = local_connections.business_id.astype('category', categories=local_businesses).cat.codes

    assert(local_col[local_col<0].shape[0] == 0)
    assert(local_row[local_row<0].shape[0] == 0)
    local_sparse_matrix = csc_matrix((local_data, (local_row, local_col)), 
                                     shape=(len(local_businesses), len(local_users)))
    print(local_sparse_matrix.shape)
    if savename is not None:
        pickle.dump(local_sparse_matrix, open('data/{}_local.pck'.format(savename), 'wb'))
    
    # Now set up tourist array
    tourist_users = set(tourist_connections.user_id)
    tourist_businesses = set(tourist_connections.business_id)
    tourist_data = tourist_connections['stars'].tolist()
    tourist_col = tourist_connections.user_id.astype('category', categories=tourist_users).cat.codes
    tourist_row = tourist_connections.business_id.astype('category', categories=tourist_businesses).cat.codes
    assert(tourist_row[tourist_row<0].shape[0] == 0)
    assert(tourist_col[tourist_col<0].shape[0] == 0)
    tourist_sparse_matrix = csc_matrix((tourist_data, (tourist_row, tourist_col)), 
                                       shape=(len(tourist_businesses), len(tourist_users)))
    print(tourist_sparse_matrix.shape)
    if savename is not None:
        pickle.dump(tourist_sparse_matrix, open('data/{}_tourist.pck'.format(savename), 'wb'))


    # Verify that the businesses are indeed the same by checking that each business id is correctly tagged with the right row for an example row
    test_ind = local_row.iloc[1] # 1 could be anything in the range of rows
    biz_ind = local_row[local_row == test_ind].index
    test_biz = local_connections.loc[biz_ind]
    test_biz = set(test_biz['business_id'])

    tour_ind = tourist_row[tourist_row == test_ind].index
    tour_biz = tourist_connections.loc[tour_ind]
    tour_biz = tour_biz['business_id']

    matches = [tour_biz.iloc[i] in test_biz for i in range(tour_biz.shape[0])]
    assert(all(matches))



    return local_sparse_matrix,  tourist_sparse_matrix


_, _ = get_city_array(cities[CITYNAME], connections, user_res, major_businesses, CITYNAME)

    
    
    
    
    
    
    
    


# In[135]:

# user_res.keys()
# LV_users = user_res[user_res['group'] == 1]
# LV_businesses = major_businesses[major_businesses['metro_area'] == 1]
# # LV_connections = connections[connections['user_id'].isin(m0_users['user_id'])]
# print(connections.shape)
# LV_connections = connections[connections['user_id'].isin(LV_users['user_id'])]
# print(LV_connections.shape)
# LV_connections = LV_connections[LV_connections['business_id'].isin(LV_businesses['business_id'])]
# print(LV_connections.shape)


# # connections.user_id

# LV_local = LV_connections.pivot(index='user_id', columns='business_id', values='stars')
# # print(LV_local)
# print(LV_local.shape)
# LV_nonnan = LV_local[np.isnan(LV_local) == False]
# # ~np.isnan(LV_local) == True
# print(LV_nonnan)
# # LV_sparse = scipy.sparse.csr_matrix(LV_local.values.T)
# # print(LV_sparse.shape)

# # c_maxes = connections.groupby(['user_id', 'business_id']).review_count.transform(max)
# # c2 = connections[connections.review_count == c_maxes]
# # # print(connections.shape)
# # # print(c2.shape)


# In[92]:



# LV_users = set(LV_connections.user_id)
# LV_businesses = set(LV_connections.business_id)

# data = LV_connections['stars'].tolist()
# col = LV_connections.user_id.astype('category', categories=LV_users).cat.codes
# row = LV_connections.business_id.astype('category', categories=LV_businesses).cat.codes
# sparse_matrix = csc_matrix((data, (row, col)), shape=(len(LV_businesses), len(LV_users)))

