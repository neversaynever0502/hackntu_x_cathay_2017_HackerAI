#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:49:50 2017

@author: kevin
"""

import re
import os
import pandas as pd

dataset_filepath = '/Users/kevin/Desktop/hackntu_x_cathay_2017-master/dataset'

cti_folder = os.path.join(dataset_filepath, "cti", "partition_time=3696969600")
cctxn_folder = os.path.join(dataset_filepath, "cctxn", "partition_time=3696969600")
atm_folder = os.path.join(dataset_filepath, "atm", "partition_time=3696969600")
mybank_folder = os.path.join(dataset_filepath, "mybank", "partition_time=3696969600")

#combime data
data_path = '/Users/kevin/Desktop/hackntu_x_cathay_2017-master/dataset/atm/partition_time=3696969600'
data_list = [os.path.join(data_path, x) for x in os.listdir(data_path) if 'csv' in x]
full_data = pd.DataFrame()
for data in data_list:
    tmp_data = pd.read_csv(data, header=None)
    full_data = full_data.append(tmp_data)
full_data = pd.DataFrame(full_data.values.tolist())

#cctxn
cctxn = pd.read_csv('/mnt/ebs/hackathon-encoded/full_cctxn.csv', index_col=0)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id', 'txn_amt', 'original_currency_code', 'consumption_category_desc',
             'mcc_code_desc', 'merchant_category_code', 'card_type', 'card_level', 'partition_time', 'theme']
old_col_name = list(cctxn.columns)
new_col_name = dict(zip(old_col_name, col_names))
cctxn.ix[:, [9, 10, 12, 13, 14]] = cctxn.ix[:, [9, 10, 12, 13, 14]].applymap(lambda x: re.findall('\w+', re.split(':', x)[-1])[0])
cctxn.ix[:, 8] = cctxn.ix[:, 8].apply(lambda x: float(re.findall('\d+\.\d+' ,re.split(':', x)[-1])[0]))
cctxn.ix[:, 11] = cctxn.ix[:, 11].apply(lambda x: '\\' + '\\'.join(re.findall('\w+', re.split(':', x)[-1])))
cctxn = cctxn.rename(columns=new_col_name)
cctxn.to_csv('/mnt/ebs/hackathon-encoded/final_cctxn.csv')

#atm
atm = pd.read_csv(atm_folder+'/part-00000-5e3d96e9-cc3a-4b47-9ef3-426efc2b2341.csv', header=None)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id', 'txn_amt', 'txn_fee_amt', 'trans_type', 'target_bank_code',
             'target_acct_nbr', 'address_zipcode', 'machine_bank_code', 'partition_time', 'theme']
old_col_name = list(atm.columns)
new_col_name = dict(zip(old_col_name, col_names))
atm.ix[:, [10, 11, 12, 14]] = atm.ix[:, [10, 11, 12, 14]].applymap(lambda x: re.findall('\w+', re.split(':', x)[-1])[0])
atm.ix[:, [8, 9]] = atm.ix[:, [8, 9]].applymap(lambda x: float(re.findall('\d+\.\d+' ,re.split(':', x)[-1])[0]))
atm.ix[:, 13] = atm.ix[:, 13].apply(lambda x: re.findall('\d+' ,re.split(':', x)[-1])[0])
atm = atm.rename(columns=new_col_name)

#cti
cti = pd.read_csv('/mnt/ebs/hackathon-encoded/full_cti.csv', index_col=0)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id', 'call_nbr', 'end_call_date', 'type_desc', 'detail_desc', 'business_desc',
             'partition_time', 'theme']
old_col_name = list(cti.columns)
new_col_name = dict(zip(old_col_name, col_names))
cti.ix[:, 8] = cti.ix[:, 8].apply(lambda x: re.findall('\w+' ,re.split(':', x)[-1])[0])
cti.ix[:, [9, 10, 11]] = cti.ix[:, [9, 10, 11]].applymap(lambda x: re.findall('\w+', re.split(':', x)[-1])[0])
cti = cti.rename(columns=new_col_name)
cti.to_csv('/mnt/ebs/hackathon-encoded/final_cti.csv')

#mybank
mybank = pd.read_csv(mybank_folder+'/part-00000-410fc821-e2c9-41f9-91bc-d66d95155f5a.csv', header=None)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id', 'amt', 'currency_code', 'partition_time', 'theme']
old_col_name = list(mybank.columns)
new_col_name = dict(zip(old_col_name, col_names))
mybank.ix[:, 8] = mybank.ix[:, 8].apply(lambda x: float(re.findall('\d+\.\d+' ,re.split(':', x)[-1])[0]))
mybank.ix[:, 9] = mybank.ix[:, 9].apply(lambda x: re.findall('\w+' ,re.split(':', x)[-1])[0])
mybank = mybank.rename(columns=new_col_name)

#customer_profile
profile = pd.read_csv('/mnt/ebs/hackathon-encoded/full_profile.csv', index_col=0)
col_names = ['customer_id', 'birth_time', 'gender', 'contact_loc', 'contact_code', 'register_loc',
             'register_code', 'start_time', 'aum', 'net_profit', 'credit_card_flag', 'loan_flag',
             'deposit_flag', 'wealth_flag', 'partition_time']
old_col_name = list(profile.columns)
new_col_name = dict(zip(old_col_name, col_names))
profile = profile.rename(columns=new_col_name)
profile.to_csv('/mnt/ebs/hackathon-encoded/final_profile.csv')

