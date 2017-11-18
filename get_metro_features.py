import pandas as pd
#import numpy as np
from scipy import cluster
import datetime

def link_for_metro(user_df, business_df, review_df):

# each row is a user review of a restaurant, columns are:
# 1) user_id 2) business_id 3) city (of business) 4) (user) review_count

    linked_review = pd.merge(review_df[['user_id','business_id','date']],
                             business_df[['business_id','city',
                                          'latitude', 'longitude']],
                             on='business_id')
    return pd.merge(linked_review,
                    user_df['user_id', 'review_count','yelping_since'],
                    on='user_id')


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
    percent_in_metro = counted.groupby('user_id')['review_count'].apply(
                                                    lambda x: x/x.sum())
    percent_in_metro = percent_in_metro.reset_index()
    percent_in_metro = percent_in_metro.rename(columns={
        'review_count':'review_metro_percent'})
    #percent_in_metro now has three columns: UID, metro_area, review_percent
    return df.merge(percent_in_metro, on=['user_id','metro_area'])
    # df now has one additional column: 'review_metro_percent'


def calc_num_weeks_metro(df):
    """
    df is a pandas dataframe where each entry corresponds to a review paired
    with the corresponding user_id.
    Df must have a field 'metro_area', which indicates
    the metro area of each review. Df must have a field 'date', which
    is modified

    Returns percent, a df that indicates what percentage of each user's reviews
    are in each metro area
    """
    week = pd.to_datetime(df['date']).dt.week
    year = pd.to_datetime(df['date']).dt.year
    df['week-year'] = week.map(str) + '-' + year.map(str)


    #number of unique reviews by a user, in a metro area, in a given (week,year)
    weeks_in_metro = df.groupby(['user_id','metro_area'])[
                                    'week-year'].nunique()

    #for each user, determine proportion of user's reviews that
    #are in each metro area
    weeks_in_metro = weeks_in_metro.rename(columns={
        'week-year':'weeks_in_metro'}).reset_index()

    df = df.merge(weeks_in_metro, on=['user_id','metro_area'])
    df = weeks_on_yelp(df)  # add 'days_on_yelp' column
    df['weeks_in_metro'] = df['weeks_in_metro'] / df['weeks_on_yelp']



def weeks_on_yelp(df):
    join_date = pd.to_datetime(df['yelping_since']).dt.date
    now_date = datetime.date(2017, 12, 1)
    df['weeks_on_yelp'] = (now_date - join_date).dt.days
    return df
    #this
    # is a
    # timedelta
    # object -> int

def num_metros_visited(df):
    df['num_metros_visited'] = df['metro_area']
    df.groupby('user_id')['num_metros_visited'].nunique()
    return df.reset_index()

def reviews_per_week_per_metro(df):
    # count up entries having same user ID and metro_area combo
    counted = df.groupby(['user_id', 'metro_area'])['review_count'].count()

    df['reviews_per_wk_per_metro'] = counted  / (df['weeks_in_metro']
                                     * df['weeks_on_yelp'])
