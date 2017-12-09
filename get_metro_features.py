import pandas as pd
#import numpy as np
from scipy import cluster
import datetime

def link_for_metro(user_df, business_df, review_df):

# users must have been edited to have a 'weeks on yelp' column and business_df must
# have been edited to have a 'metro_area column'
    if 'weeks_on_yelp' not in user_df.keys():
        raise(ValueError('user_df must contain \'weeks_on_yelp\' column'))
    if 'metro_area' not in business_df.keys():
        raise(ValueError('business_df must contain \'metro_area\' column'))
# each row is a user review of a restaurant, columns are:
# 1) user_id 2) business_id 3) city (of business) 4) (user) review_count

    linked_review = pd.merge(review_df[['user_id','business_id','date', 'stars']],
                             business_df[['business_id','metro_area',
                                          'latitude', 'longitude']],
                             on='business_id', how='inner')
    return pd.merge(linked_review,
                    user_df[['user_id', 'review_count','yelping_since']],
                    on='user_id', how='inner')


def add_metro_area(df, K=12):
    _, df['metro_area'] = cluster.vq.kmeans2(df, K)
    return df

def calc_perc_metro(df):
    """
    df is a pandas dataframe where each entry corresponds to a review paired
    with the corresponding user_id.
    Df must have a field 'metro_area', which indicates
    the metro area of each review. Df must have a field 'review_count', which
    is modified

    Returns percent, a df that indicates what percentage of each user's reviews
    are in each metro area
    """

    #count up entries having same user ID and metro_area combo
    counted = df.groupby(['user_id','metro_area'])['review_count'].count()
    #for each user, determine proportion of user's reviews that
    #are in each metro area
    percent_in_metro = counted.groupby(level=0).apply(lambda x: x/float(x.sum()))
    percent_in_metro = percent_in_metro.reset_index()
    percent_in_metro = percent_in_metro.rename(columns={
        'review_count':'review_metro_percent'})
    #percent_in_metro now has three columns: UID, metro_area, review_percent
    return pd.merge(df, percent_in_metro, on=['user_id','metro_area'], how='inner')
    # df now has one additional column: 'review_metro_percent'


def calc_num_weeks_metro(df):
    """
    df is a pandas dataframe where each entry corresponds to a review paired
    with the corresponding user_id.
    Df must have a field 'metro_area', which indicates
    the metro area of each review. Df must have a field 'date', which
    is modified

    """
    week = pd.to_datetime(df['date']).dt.week
    year = pd.to_datetime(df['date']).dt.year
    df['week-year'] = week.map(str) + '-' + year.map(str)


    #number of unique reviews by a user, in a metro area, in a given (week,year)
    weeks_in_metro = df.groupby(['user_id','metro_area'])[
                                    'week-year'].nunique()

    #for each user, determine proportion of user's reviews that
    #are in each metro area
    weeks_in_metro = weeks_in_metro.reset_index().rename(columns={
        'week-year':'weeks_in_metro'})

    df = pd.merge(df, weeks_in_metro, on=['user_id','metro_area'], how='inner')
    df = weeks_on_yelp(df)  # add 'days_on_yelp' column
    # print('weeks_in_metro' in df.keys())
    df['weeks_in_metro'] = df['weeks_in_metro'] / df['weeks_on_yelp']
    return df


def weeks_on_yelp(df):
    join_date = pd.to_datetime(df['yelping_since']).dt.date
    now_date = datetime.date(2017, 12, 1)
    df['weeks_on_yelp'] = (now_date - join_date).dt.days
    return df
    #this
    # is a
    # timedelta
    # object -> int

# def num_metros_visited(df):
#     df['num_metros_visited'] = df['metro_area']
#     dnew['num_metros_visited'] = df.groupby('user_id')['num_metros_visited'].nunique()
#     dnew.reset_index()
    # return dnew

def num_metros_visited(df):
    num_visited = df.groupby('user_id')['metro_area'].nunique().reset_index()
    num_visited = num_visited.rename(columns={
                                     'metro_area':'num_metros_visited'})
    print('Grouped. Now merge')
    return df.merge(num_visited, on='user_id')


def reviews_per_week_per_metro(df):
    # count up entries having same user ID and metro_area combo
    counted = df.groupby(['user_id', 'metro_area'])['review_count'].count()
    print(counted.shape)
    counted = counted.reset_index()
    print(counted.shape, df.shape)
    reviews_per_wk_per_metro = counted  / (df['weeks_in_metro']
                                     * df['weeks_on_yelp'])
    reviews_per_wk_per_metro.rename(columns={'review_count': 'reviews_per_wk_per_metro'}).reset_index()
    reviews_per_wk_per_metro
    print(reviews_per_wk_per_metro.keys())
    print('Grouped. ')
    return df.merge(reviews_per_wk_per_metro, on=['user_id','metro_area'])


def define_user_features(user_df, feature_df, num_metro_areas):
    """
        Given a DataFrame of users and a dataframe of features this function creates a feature vector for the user at each metro area
    """
    user_features = user_df[['user_id']].copy()
    user_features = pd.merge(user_features, feature_df[['user_id', 'num_metros_visited', 'weeks_on_yelp']].copy(), on='user_id', how='left')
    print(user_features.shape)


    for m in range(num_metro_areas):
        print('evaluating metro: {}'.format(m))
        # Get percentage of reviews in each metro
        temp_perc = feature_df.query('metro_area == {}'.format(m)).groupby(['user_id'])['review_metro_percent'].first().reset_index()
        user_features = pd.merge(user_features, temp_perc, on='user_id', how='left')
        user_features = user_features.rename(columns={'review_metro_percent':'m{}_percent'.format(m)})
        
        # Get review weeks in each metro
        temp_weeks = feature_df.query('metro_area == {}'.format(m)).groupby(['user_id'])['weeks_in_metro'].first().reset_index()
        user_features = pd.merge(user_features, temp_weeks, on='user_id', how='left')
        user_features = user_features.rename(columns={'weeks_in_metro':'m{}_weeks'.format(m)})
        print(user_features.shape)
    return user_features







