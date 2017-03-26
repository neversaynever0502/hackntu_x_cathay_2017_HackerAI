#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:53:35 2017

@author: kevin
"""

import re
import os
import pandas as pd

#original-eg
ori_eg = pd.read_csv('/mnt/ebs/hackathon-encoded/cctxn/partition_time=3696969600/part-00000-b78d8440-c555-42a1-a04a-91b086d9a4c6.csv').ix[:2, :]

#integrate all data to one set
main_path = '/mnt/ebs/hackathon-encoded/cctxn'
sub_path = [os.path.join(main_path, x) for x in os.listdir(main_path) if 'partition' in x]

full_data = pd.DataFrame()
for data_path in sub_path:
 	data_list = [os.path.join(data_path, y) for y in os.listdir(data_path)]
 	for data in data_list:
 		tmp_data = pd.read_csv(data, header=None)
     	full_data = full_data.append(tmp_data)
full_data = pd.DataFrame(full_data.values.tolist())
full_data.to_csv('/mnt/ebs/hackathon-encoded/full_cctxn_test.csv')


#data processed
cctxn = pd.read_csv('/mnt/ebs/hackathon-encoded/cctxn.csv', index_col=0)
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


import re
import os
import pandas as pd

 main_path = '/mnt/ebs/hackathon-encoded/atm'
 sub_path = [os.path.join(main_path, x) for x in os.listdir(main_path) if 'partition' in x]

 full_data = pd.DataFrame()
 for data_path in sub_path:
 	data_list = [os.path.join(data_path, y) for y in os.listdir(data_path)]
 	for data in data_list:
 		tmp_data = pd.read_csv(data, header=None)
     	full_data = full_data.append(tmp_data)
 full_data = pd.DataFrame(full_data.values.tolist())
 full_data.to_csv('/mnt/ebs/hackathon-encoded/full_atm.csv')


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

mybank = pd.read_csv('/mnt/ebs/hackathon-encoded/full_mybank.csv', index_col=0)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id','fee', 'amt', 'currency_code', 'partition_time', 'theme']
old_col_name = list(mybank.columns)
new_col_name = dict(zip(old_col_name, col_names))
mybank.ix[:, 8] = mybank.ix[:, 8].apply(lambda x: re.split(':', x)[-1])
mybank.ix[:, 9] = mybank.ix[:, 9].apply(lambda x: re.split(':', x)[-1])
mybank.ix[:, 10] = mybank.ix[:, 10].apply(lambda x: re.findall('\w+' ,re.split(':', x)[-1])[0])
mybank = mybank.rename(columns=new_col_name)
mybank.to_csv('/mnt/ebs/hackathon-encoded/final_mybank.csv')


atm = pd.read_csv('/mnt/ebs/hackathon-encoded/full_atm.csv', index_col=0)
col_names = ['actor_type', 'actor_id', 'action_type', 'action_time', 'object_type', 'object_id',
             'channel_type', 'channel_id', 'txn_amt', 'txn_fee_amt', 'trans_type', 'target_bank_code',
             'target_acct_nbr', 'address_zipcode', 'machine_bank_code', 'partition_time', 'theme']
old_col_name = list(atm.columns)
new_col_name = dict(zip(old_col_name, col_names))
atm.ix[:, [10, 11, 12, 14]] = atm.ix[:, [10, 11, 12, 14]].applymap(lambda x: re.findall('\w+', re.split(':', x)[-1])[0])
atm.ix[:, [8, 9]] = atm.ix[:, [8, 9]].applymap(lambda x: float(re.findall('\d+\.\d+' ,re.split(':', x)[-1])[0]))
atm.ix[:, 13] = atm.ix[:, 13].apply(lambda x: re.findall('\d+' ,re.split(':', x)[-1])[0])
for i in range(len(atm)):
    try:
        atm.ix[i, 13] = map(lambda x: re.findall('\d+' ,re.split(':', x)[-1])[0], atm.ix[i, 13])
    except:
        atm.ix[i, 13] = 'NaN'

atm = atm.rename(columns=new_col_name)






