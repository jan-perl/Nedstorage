# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#samenvatting van NED data
# -

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import subprocess
import requests
import json
import os
import time

myapikey = subprocess.getoutput("cat ../data/ned-api.key")
print(myapikey)
baseurl='https://api.ned.nl/v1'

gasmaand_l= pd.read_csv('../data/all-consuming-gas-afgelopen-12-maanden.csv')
gasmaand_l

# +
#sns.lineplot(gasmaand_l)

# +
urlhost = "https://api.ned.nl" 
baseurl = urlhost+"/v1/utilizations"
url=baseurl

headers = {
 'X-AUTH-TOKEN': myapikey,
 'accept': 'application/ld+json'}
params = {'point': 0, 'type': 2, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2024-11-17', 'validfrom[after]': '2024-11-16'}
response = requests.get(url, headers=headers, params=params, allow_redirects=False)

r2=re.sub( '],"hydra:view.*$',']}',re.sub('.@context.*hydra:member.','"member"',response.text) )
#print(r2)
# -

vpf=json.loads(response.text)
print(vpf)

print (vpf['hydra:member'])

print (pd.json_normalize(vpf['hydra:view']))

#vpd= pd.read_json("{'member':"+vpf['hydra:member']+"}")
pval=pd.json_normalize(vpf['hydra:member'])
pval

energytypes=pd.read_csv("../data/energytypes.txt",sep=" ").set_index('index')
energytypes
energytypes_dict=energytypes.to_dict()['Energytype']
energytypes_dict


# +
#parameters: zie https://ned.nl/nl/handleiding-api
#activity 1 Providing,2 Consuming,3 Import, 4 Export
def getnedvals(params,ptypes,activity):
    eaders = {
 'X-AUTH-TOKEN': myapikey,
 'accept': 'application/ld+json'}
    rlist=[];
    for ptype in ptypes:
        params['type']=ptype
        params['activity']=activity
        getmore=True
        url=baseurl
        while getmore:
            response = requests.get(url, headers=headers, params=params, allow_redirects=False)
            if not response:
                print (params)
                raise Exception(f"Non-success status code: {response.status_code}")
            vpf=json.loads(response.text)

            pval=pd.json_normalize(vpf['hydra:member'])
            if (pval.size ==0):
                print('No values for ' +str(ptype))
            elif (pval.columns[0]=='member'):
                print('No values for ' +str(ptype))
            else:
                pval['energytypenr']=ptype
                pval['energytype']=str(ptype)+ " - "+ energytypes_dict[ptype]
                rlist.append(pval)
            pval=pd.concat(rlist)
            getmore ='hydra:next' in vpf['hydra:view']
            if getmore:
#                print (vpf['hydra:view'])
                getmore = url != vpf['hydra:view']['hydra:last']
                url= urlhost+vpf['hydra:view']['hydra:next']      
    for ccol in ['validfrom' , 'validto', 'lastupdate']:
        pval[ccol]=pd.to_datetime(pval[ccol])
    return pval

params1= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2025-09-15', 'validfrom[after]': '2025-09-01'}

pset1b= getnedvals(params1,[0,1],1)
# -

pset1typs=[0,1,2,17,18,19]
pset1= getnedvals(params1,pset1typs,1)

pset1.dtypes

#vergelijk waarden https://co2monitor.nl/energiebronnen/terugblik# in MW
pset1kw=pset1.copy()
pset1kw['volume']=pset1kw['volume']/1000
sns.lineplot(data=pset1kw,x="validfrom",y="volume",hue="energytype")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

pset2typs=[23,31,53,54,55,56]
pset2= getnedvals(params1,pset2typs,2)
pset2kw=pset2.copy()
pset2kw['volume']=pset2kw['volume']/1000
sns.lineplot(data=pset2kw,x="validfrom",y="volume",hue="energytype")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# +
def mkrestcat(df,defs,rtype):
    allcols=set(df.columns)
    summcols=allcols-set(["@id","id","type","volume","capacity","percentage","emission",
                          "energytype","energytypenr","emissionfactor","lastupdate","activity"])
    print(summcols)
    dfcalc=df.copy()
    mmap= dfcalc['energytypenr'].map(defs)
    
    dfcalc['volume']=(mmap)*dfcalc['volume']
    dfcalc['capacity']=(mmap)*dfcalc['capacity']
    dfcalc['emission']=(mmap)*dfcalc['emission']
    dfcalc=dfcalc[False==pd.isna(mmap) ]    

    dfcalc=dfcalc.groupby( list(summcols) ).agg('sum').reset_index()
    dfcalc['energytypenr']=rtype
    dfcalc['energytype']=str(rtype)+ " - "+ energytypes_dict[rtype]
    return dfcalc
    
#check waarde gemaakt in 26 om verschil te begrijpen
pset2r=mkrestcat(pset2,{23:1,54:-0.2,55:-1},26)
sns.lineplot(data=pd.concat([pset2r,pset2]),x="validfrom",y="volume",hue="energytype")    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# -

#restcat is som van met niet-duurzame middelen opgewekt
pset1r=mkrestcat(pset1,{0:1,1:-1,2:-1,17:-1},10)
#print(pset1r)
#waarde om rest opwek te begrijpen
sns.lineplot(data=pd.concat([pset1r,pset1]),x="validfrom",y="volume",hue="energytype") 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

if False:
    params3= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 3,
 'validfrom[strictly_before]': '2025-09-15', 'validfrom[after]': '2025-09-01'}

    pset3= getnedvals(params3,[0,23],3)
    pset3kw=pset3.copy()
    pset3kw['volume']=pset3kw['volume']/1000
    sns.scatterplot(data=pset3kw,x="validfrom",y="volume",hue="energytype")

if False:
    params4= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 4,
     'validfrom[strictly_before]': '2025-09-15', 'validfrom[after]': '2025-09-01'}

    pset4= getnedvals(params4,[0,23],3)
    pset4kw=pset4.copy()
    pset4kw['volume']=pset4kw['volume']/1000
    sns.scatterplot(data=pset4kw,x="validfrom",y="volume",hue="energytype")
pset1t=mkrestcat(pd.concat ( [pset1,pset2]) ,{0:1,23:1},10)
pset1tkw=pset1t.copy()
pset1tkw['volume']=pset1tkw['volume']/1000
sns.lineplot(data=pset1tkw,x="validfrom",y="volume",hue="energytype")  
# +
#now get full years

# +
params1y= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2024-12-31', 'validfrom[after]': '2024-01-01'}

yset1= getnedvals(params1y,pset1typs,1)
# -

#when running full workbook, don't overload server
time.sleep(360)

yset2= getnedvals(params1y,pset2typs,2)

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr=pd.concat ( [yset1,yset2]) 
egasyr.to_pickle("../intermediate/egasyr2024.pkl")

# +
params1y23= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2023-12-31', 'validfrom[after]': '2023-01-01'}

yset231= getnedvals(params1y23,pset1typs,1)
# -

#when running full workbook, don't overload server
time.sleep(360)

yset232= getnedvals(params1y23,pset2typs,2)

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr23=pd.concat ( [yset231,yset232]) 
egasyr23.to_pickle("../intermediate/egasyr2023.pkl")

time.sleep(360)

# +
params1y25= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2025-12-31', 'validfrom[after]': '2025-01-01'}

yset251= getnedvals(params1y25,pset1typs,1)
# -

time.sleep(360)

yset252= getnedvals(params1y25,[23,31],2)
#for 2022: No values for 53, No values for 54, No values for 55, No values for 56

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr25=pd.concat ( [yset251,yset252]) 
egasyr25.to_pickle("../intermediate/egasyr2025.pkl")

time.sleep(360)

# +
params1y22= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2022-12-31', 'validfrom[after]': '2022-01-01'}

yset221= getnedvals(params1y22,pset1typs,1)
# -

time.sleep(360)

yset222= getnedvals(params1y22,[23,31],2)
#for 2022: No values for 53, No values for 54, No values for 55, No values for 56

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr22=pd.concat ( [yset221,yset222]) 
egasyr22.to_pickle("../intermediate/egasyr2022.pkl")

time.sleep(360)

# +
params1y26= {'point': 0, 'type': 0, 'granularity': 5, 'granularitytimezone': 1, 'classification': 2, 'activity': 1,
 'validfrom[strictly_before]': '2026-12-31', 'validfrom[after]': '2026-01-01'}

yset261= getnedvals(params1y26,pset1typs,1)
# -

time.sleep(360)

yset262= getnedvals(params1y26,[23,31],2)
#for 2022: No values for 53, No values for 54, No values for 55, No values for 56

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr26=pd.concat ( [yset261,yset262]) 
egasyr26.to_pickle("../intermediate/egasyr2026.pkl")

time.sleep(360)
