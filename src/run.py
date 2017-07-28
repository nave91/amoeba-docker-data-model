# import necessary packages
import shapley_value
import pandas as pd
import numpy as np
import sqlalchemy
import itertools
import re
import os
import datetime

# connect to the database
try:
    conn = sqlalchemy.create_engine(os.environ['GOSPEL_DB_URL'])
except:
    print ("Unable to connect to the database")

# get the customers' booking journey
journey = pd.read_sql('SELECT channel_journey, market_id, AVG(revenue) AS avg_of_revenue_from_channel_journey FROM (SELECT DISTINCT user_id, market_id, revenue_top_line_revenue AS revenue, string_agg(DISTINCT channel, \',\' ORDER BY channel) AS channel_journey FROM zzz_dbt_test_abraar.touches_interpolated GROUP  BY 1, 2, 3) a GROUP BY channel_journey, market_id', conn)
rev_journey = journey.dropna()

def channel_and_revs(df=mrkt_rev_journey, c_one='channel_journey', c_two='avg_of_revenue_from_channel_journey'):
    """
    function 1: get all channels and average revenue
    """
    chs = np.array(df[c_one])
    mon = np.array(df[c_two])
    chs_ = [len(chs[n].split(',')) for n in list(range(len(chs)))]
    channels = np.array(chs)[np.argsort(chs_)].tolist()
    money_from_channels = np.array(mon)[np.argsort(chs_)].tolist()
    return channels, money_from_channels

def s_channels(lst=channels):
    """
    function 2: get single channels and update channels list
    """
    single_channels = []
    for n in list(range(len(lst))):
        for m in list(range(len(lst[n].split(',')))):
            if len(lst[n].split(',')) == 1:
                single_channels.append(lst[n].split(','))
            for p in list(range(len(single_channels))):
                if lst[n].split(',')[m] not in single_channels[p]:
                    lst.append(lst[n].split(',')[m])
    return lst

def zerolistmaker(n):
    """
    function 3: make list of zeros
    """
    listofzeros = [0] * n
    return listofzeros

def sing(lst=channels):
    """
    function 4: get only singles that are ordered
    """
    singles = []
    for n in list(range(len(lst))):
        if len(lst[n].split(',')) == 1:
            singles.append(lst[n])
    singles.sort()
    return singles

def comb(lst=singles):
    """
    function 5: get all combinations of the singles
    """
    combs = []
    for n in list(range(2, len(lst)+1)):
        for x in itertools.combinations(lst, n):
            combs.append(list(x))
    newcombs = [','.join(m) for m in combs]
    lst.extend(newcombs)
    return lst

def comp(lst_one=singles, lst_two=channels):
    """
    function 6: compare and extend channels
    """
    if set(lst_one) != set(lst_two):
        lst = np.setdiff1d(lst_one, lst_two)
        lst_two.extend(lst)
    return lst_two

def rev_comp(lst_one=channels, lst_two=money_from_channels):
    """
    function 7: compare and extend revenue from channels
    """
    if len(lst_one) != len(lst_two):
        diff = len(lst_one) - len(lst_two)
        zeros = zerolistmaker(diff)
        lst_two.extend(zeros)
    return lst_two

def mon_channels(lst_one=channels, lst_two=money_from_channels):
    """
    function 8: get actual money from channels
    """
    for n in list(range(len(lst_one))):
        for m in list(range(len(lst_one))):
            if lst_one[n] != lst_one[m]:
                if lst_one[n] in lst_one[m]:
                    lst_two[m] += lst_two[n]
    return lst_two

def pretty_ev(lst_one=channels, lst_two=money_from_channels):
    """
    fuction 9: pretty up everything
    """
    _chs = np.array(lst_one)
    _mon = np.array(lst_two)
    _chs_ = [len(_chs[n].split(',')) for n in list(range(len(_chs)))]
    channels_att = np.array(_chs)[np.argsort(_chs_)].tolist()
    money_from_channels_att = np.array(_mon)[np.argsort(_chs_)].tolist()
    return channels_att, money_from_channels_att

def s_c(lst=channels):
    """
    function 10: single channels for Shapley comparison
    """
    single_chs = []
    for n in list(range(len(lst))):
        if len(lst[n].split(',')) == 1:
            single_chs.append(lst[n])
    single_chs.sort()
    return single_chs

def shapley_computation(lst_one=channels_att, lst_two=money_from_channels_att, lst_three=single_chs):
    """
    function 11: Shapley computation
    """
    dictionary_att = dict(zip(lst_one, lst_two))
    check = shapley_value.characteristic_function_check(lst_three, dictionary_att)
    shapley_v = shapley_value.Coop_Game(lst_three, dictionary_att).shapley()
    return shapley_v

def shap_intro(diction=shapley_v, mrkt_id=x):
    """
    function 12: Dict to DataFrame
    """
    df = pd.DataFrame.from_dict(shapley_v, orient='index').rename(columns={0:'average_contribution'})
    df['market_id'] = mrkt_id
    df['date'] = datetime.datetime.utcnow()
    return df

# Compute shapley value, append to DataFrame, insert into table
for x in list(np.unique(rev_journey['market_id'])):
    mrkt_rev_journey = rev_journey[rev_journey['market_id'] == x]
    channels, money_from_channels = channel_and_revs(df=mrkt_rev_journey, c_one='channel_journey', c_two='avg_of_revenue_from_channel_journey')
    s_channels(lst=channels)
    sing(lst=channels)
    comb(lst=singles)
    comp(lst_one=singles, lst_two=channels)
    rev_comp(lst_one=channels, lst_two=money_from_channels)
    mon_channels(lst_one=channels, lst_two=money_from_channels)
    channels_att, money_from_channels_att = pretty_ev(lst_one=channels, lst_two=money_from_channels)
    s_c(lst=channels)
    shapley_computation(lst_one=channels_att, lst_two=money_from_channels_att, lst_three=single_chs)
    shap_intro(diction=shapley_v, mrkt_id=x)
    df.to_sql(name='fair_attribution', con=conn, schema='DOCKER_ML_OUTPUT_SCHEMA', if_exists='append')
