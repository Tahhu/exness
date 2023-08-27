import numpy as np
import pandas as pd
from itertools import product
import lightgbm as lgb
from lightgbm import Dataset
from typing import List


def create_train_df(df: pd.DataFrame, 
                    START_PREDICT_DATE: str,
                    END_PREDICT_DATE: str
                   ):
    
    START_TRAIN_DATE = pd.to_datetime('2016-01-02')
    START_PREDICT_DATE = pd.to_datetime(START_PREDICT_DATE)
    END_PREDICT_DATE = pd.to_datetime(END_PREDICT_DATE)
    
    range_date = pd.date_range(START_TRAIN_DATE, END_PREDICT_DATE, freq='D')
    unique_shop_id = df['id'].unique().tolist()
    
    train_grid = pd.DataFrame(list(product(unique_shop_id,
                                           range_date)),
                              columns = ['id', 'dt']
                             )
    df_train = train_grid.merge(df, how = 'left', on = ['id', 'dt'])
    df_train['sales'] = df_train['target'].copy()
    df_train['target'] = df_train['target'].fillna(0)
    df_train['split'] = np.where(df_train['dt'] >= START_PREDICT_DATE, 'test', 'train')
    df_train.loc[df_train['dt'] >= START_PREDICT_DATE, 'target'] = np.nan 

    return df_train

def create_calendar_vars(df, date_col):
    """
    Create calendar vars
    """

    def week_of_month(dt):
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return int(np.ceil(adjusted_dom / 7.0))

    df_ = df.copy()
    df_['year'] = df_[date_col].dt.year.astype('int')
    df_['quarter'] = df_[date_col].dt.quarter.astype('int')
    df_['month'] = df_[date_col].dt.month.astype('int')
    df_['dayofweek'] = df_[date_col].dt.dayofweek.astype('int')
    df_['dayofyear'] = df_[date_col].dt.dayofyear.astype('int')
    df_['dayofmonth'] = df_[date_col].dt.day.astype('int')
    df_['week'] = df_[date_col].dt.isocalendar().week.astype('int') 
    df_['week'] = df_['week'].astype('int')
    df_['is_weekend'] = pd.Series(df_[date_col]).apply(lambda x : x.weekday() in [5, 6]).values 
    df_['is_weekend'] = df_['is_weekend'].astype('category')
    df_['week_of_month'] = pd.Series(df_[date_col]).apply(week_of_month).values
    df_['week_of_month'] = df_['week_of_month'].astype('int')
    df_['first_day_month'] = df_['dayofmonth'].apply(lambda x: x == 1).astype('category')
    df_['second_day_month'] = df_['dayofmonth'].apply(lambda x: x == 2).astype('category')
    df_['is_evenweek'] = df_['week'] // 2 == 0 
    df_['is_evenweek'] = df_['is_evenweek'].astype('category')
    df_['week_of_month_week'] = df_['week_of_month'].astype('str') + '-'  + df_['week'].astype('str')
    df_['week_of_month_week'] = df_['week_of_month_week'].astype('category')
    df_['week_quarter'] = df_['week'].astype('str') + '-'  + df_['quarter'].astype('str') 
    df_['week_quarter'] = df_['week_quarter'].astype('category')
    df_['dayofmonth_quarter'] = df_['dayofmonth'].astype('str') + '-'  + df_['quarter'].astype('str')
    df_['dayofmonth_quarter'] = df_['dayofmonth_quarter'].astype('category')
    df_['is_evenweek_dayofweek'] = df_['is_evenweek'].astype('str') + '-'  + df_['dayofweek'].astype('str')
    df_['is_evenweek_dayofweek'] = df_['is_evenweek_dayofweek'].astype('category')
    return df_

def create_lags(df, group_columns: List[str], target_column: str, lags_range: list):
    """
    Create lags
    """
    df_ = df.copy()
    for lag in lags_range:
        df_[f'Lags_{lag}'] = df_.groupby(group_columns)[target_column].shift(lag)
        
    return df_


def create_group_means(df, target_column: str):
    """
    Create grouped means
    """
    df_ = df.copy()
    df_['mean_id'] = df_.groupby(['id'])[target_column].transform(lambda x: x.mean())
    df_['mean_year'] = df_.groupby(['year'])[target_column].transform(lambda x: x.mean())
    df_['mean_quarter'] = df_.groupby(['quarter'])[target_column].transform(lambda x: x.mean())
    df_['mean_week'] = df_.groupby(['week'])[target_column].transform(lambda x: x.mean())
    df_['mean_dayofweek'] = df_.groupby(['dayofweek'])[target_column].transform(lambda x: x.mean())
    df_['mean_dayofweek_even_week'] = df_.groupby(['dayofweek', 'is_evenweek'])[target_column].transform(lambda x: x.mean())
    
    df_['mean_dayofweek_even_id'] = df_.groupby(['dayofweek', 'is_evenweek', 'id'])['target'].transform(lambda x: x.mean())
    df_['max_dayofweek_even_id'] = df_.groupby(['dayofweek', 'is_evenweek', 'id'])['target'].transform(lambda x: x.max())
    df_['min_dayofweek_even_id'] = df_.groupby(['dayofweek', 'is_evenweek', 'id'])['target'].transform(lambda x: x.min())
    df_['diff_dayofweek_even_id'] = df_.groupby(['dayofweek', 'is_evenweek', 'id'])['target'].transform(lambda x: x.max() - x.min())
    
    df_['mean_dayofweek_week_of_month_id'] = df_.groupby(['dayofweek', 'week_of_month', 'id'])['target'].transform(lambda x: x.mean())
    df_['max_dayofweek_week_of_month_id'] = df_.groupby(['dayofweek', 'week_of_month', 'id'])['target'].transform(lambda x: x.max())
    df_['min_dayofweek_week_of_month_id'] = df_.groupby(['dayofweek', 'week_of_month', 'id'])['target'].transform(lambda x: x.min())
    df_['diff_dayofweek_week_of_month_id'] = df_.groupby(['dayofweek', 'week_of_month', 'id'])['target'].transform(lambda x: x.max() - x.min())

                                                   
    return df_


def create_rollings_means(df, target_column: str, horizon=None):
    """
    Create rolling means
    """
    
    df_ = df.copy()
    df_['rolling_mean_id'] = df_.groupby(['id'])[target_column].transform(
        lambda s: s.shift(horizon).rolling(window=7, min_periods=1, center=False).agg('mean'))
    df_['rolling_mean_id_28'] = df_.groupby(['id'])[target_column].transform(
        lambda s: s.shift(horizon).rolling(window=28, min_periods=1, center=False).agg('mean'))
        
    return df_


def create_seasonal_vars(df: pd.DataFrame, group_columns: List[str], target_column=str, horizon=None):
    """
    Create seasonal features
    """
    train = df[df['split'] == 'train']
    train['trend'] = train.groupby(group_columns)[target_column].transform(
        lambda s: s.shift(horizon).rolling(window=500, min_periods=30, center=True).agg('mean')
    )
    train['trend'] = train['trend'].fillna(method='bfill')
    train["detrended"] = train[target_column] - train["trend"]
    
    coef = train.groupby(group_columns, as_index = False)[['detrended']].mean()
    coef = coef.rename(columns = {'detrended': 'seasonal'})
    
    # Клипануть сезонные коэффициенты
    
    return coef

def filter_dataset(df):
    mask_na = (df['dt'] >= pd.to_datetime('2016-07-01'))\
                & (df['dt'] < pd.to_datetime('2017-01-01'))\
                & (df['id'].isin([4, 8, 10, 13, 17])
                  )
    early_mask = (df['dt'] <= pd.to_datetime('2016-03-01')) 
    mask = mask_na + early_mask
    
    return df[~mask]

def get_valid_index(train: 
                    pd.DataFrame, 
                    valid_size: int = 0.1,
                    random_seed = 0
                   ):
    """
    get valid index and drop them form train
    """
    np.random.seed(random_seed)
    valid_size = np.ceil(len(train) * valid_size).astype(int)
    valid_idx = np.random.choice(train.index, valid_size, replace = False)
    valid = train.loc[valid_idx]
    
    train.drop(valid_idx, axis = 0, inplace = True)
    
    return valid

def prepare_datesets(df, features, valid_size, random_seed):
    """
    Prepare dataset for lgbm
    
    """
    
    train = df[(df['split'] == 'train')]
    test = df[df['split'] == 'test']
    valid = get_valid_index(train,
                           valid_size,
                           random_seed,
                       )
    
    cat_columns = train[features].select_dtypes(include='category').columns.tolist()
    
    # lgbm datasets
    train_ds = lgb.Dataset(train[features],
                       train['target'],
                       categorical_feature=cat_columns,
                       )
    
    val_ds = lgb.Dataset(valid[features],
                         valid['target'],
                         categorical_feature=cat_columns,
                         reference=train_ds
                        )
    
    return train_ds, val_ds

def get_predictions(df, model):
    df['prediction'] = model.predict(df[model.feature_name()])
    df['prediction'] = np.clip(df['prediction'], a_min=0, a_max = None)
    
    # post_processsing
    claster = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]
    claster_holidays = [pd.to_datetime('2017-06-04')]
    all_holidays = [pd.to_datetime('2017-05-01'),
                    pd.to_datetime('2017-05-14'),
                    pd.to_datetime('2017-05-25'),
                    pd.to_datetime('2017-04-03'),
                    pd.to_datetime('2017-04-06'),
                   ]

    for shop in claster:
        df.loc[(df['id'] == shop)
              & (df['dt'].isin(claster_holidays)), 
               'prediction'
              ] = 0

    df.loc[(df['dt'].isin(all_holidays)), 
           'prediction'
          ] = 0    
        
    return df