import streamlit as st
import pandas as pd
import datetime
import calendar
import lightgbm as lgb
from lightgbm import Dataset
from scripts import *


VALID_SIZE = 0.2

data = pd.read_csv('data/train.csv')
data['dt'] = pd.to_datetime(data['dt'])


stores = st.sidebar.multiselect(
    'Select stores to get prediction', data['id'].unique())

st.title('Sales Prediction')
st.subheader('Service will provide forecasts for stores for month ahead.')

date = st.date_input("Choose date for prediction", 
                     datetime.date(2017, 7, 1), 
                     min_value=datetime.date(2017, 1, 1),
                     max_value=datetime.date(2017, 7, 1),
                 )

beg_date = datetime.datetime(date.year, date.month, 1)
n_days_in_that_month = calendar.monthrange(beg_date.year, beg_date.month)[1]
end_date = datetime.datetime(beg_date.year, beg_date.month, n_days_in_that_month)


st.write(f'Getting predictions for {date}')
st.write(f'Prediction period {beg_date} -- {end_date}')


# prepare dataset for learning
df = create_train_df(data, 
                     START_PREDICT_DATE=beg_date,
                     END_PREDICT_DATE=end_date
                    )

# create calendar features
df = create_calendar_vars(df, date_col='dt')

# create lags
df = create_lags(df, 
                 group_columns=['id'], 
                 target_column='target',
                 lags_range = [28, 29, 30, 31, 62, 91, 122, 365])

# gropued means
df = create_group_means(df, 
                        target_column='target')

# rolling means
df = create_rollings_means(df, 
                           target_column='target', 
                           horizon=n_days_in_that_month)

# seasonal coef
seasona_coef = create_seasonal_vars(df, 
                                    group_columns=['dayofweek', 'is_evenweek', 'id'], 
                                    target_column='target', 
                                    horizon=n_days_in_that_month)

df = df.merge(seasona_coef, 
              on = ['dayofweek', 'is_evenweek', 'id'], 
              how = 'left')

# filter dataset
df = filter_dataset(df)

# define features for training
features2train = ['id', 
                  'year', 'quarter', 'month',
                  'dayofmonth', 
                  'dayofweek', 
                  'week', 'is_weekend',
                  'first_day_month', 'second_day_month',
                  'week_of_month_week',
                  'mean_id',
                  'mean_year',
                  'mean_quarter',
                  'mean_week', 
                  'mean_dayofweek',
                  'mean_dayofweek_even_week',
                  'rolling_mean_id',
                  'rolling_mean_id_28',
                  'mean_dayofweek_even_id',
                  'max_dayofweek_even_id', 
                  'min_dayofweek_even_id',
                  'diff_dayofweek_even_id',     
                  'mean_dayofweek_week_of_month_id',
                  'max_dayofweek_week_of_month_id',
                  'min_dayofweek_week_of_month_id',
                  'diff_dayofweek_week_of_month_id',
                  'Lags_91', 
                  'Lags_29', 
                  'Lags_62', 
                  'Lags_122',
                  'seasonal',
                 ]

# prepeare lgbm dataset
train_ds, val_ds = prepare_datesets(df, 
                                    features2train, 
                                    valid_size=VALID_SIZE, 
                                    random_seed=0)


# boosting hyperparameters
params = {
    'objective': 'mae', 
    'eta': 0.01, # 0.1
    'max_depth' : -1,
    'seed' : 42, 
    'verbose' : -1,
     'num_threads': 8,
    
}

# train
model = lgb.train(params,
                  train_ds,
                  num_boost_round = 1000,
                  valid_sets=[train_ds, val_ds],
                  verbose_eval=500,
                  evals_result=None,
                  )
    
# prediction
df = get_predictions(df, model)


#
if stores:
    tmp_df = df[(df['id'].isin(stores))
                & (df['dt'] >=pd.to_datetime(beg_date))
               ]
    
    
else:
     tmp_df = df[(df['dt'] >=pd.to_datetime(beg_date))
               ]

    
tmp_df_grouped = pd.DataFrame(tmp_df.groupby(['dt']).agg({'prediction': 'sum'}))
tmp_df_grouped.columns = ['prediction']

st.subheader(f'Forecast for Store № {stores}')
st.line_chart(tmp_df_grouped)
