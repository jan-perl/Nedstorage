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
import matplotlib as plt
import subprocess
import requests
import json
import os

energytypes=pd.read_csv("../data/energytypes.txt",sep=" ").set_index('index')
energytypes
energytypes_dict=energytypes.to_dict()['Energytype']
energytypes_dict

#sla gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr=pd.read_pickle("../intermediate/egasyr2024.pkl")
#egasyr.dtypes
egasyr

maxd="2024-01-16T00:00:00+00:00"
psett=egasyr[(egasyr['validfrom']< pd.to_datetime(maxd))]
#psett=egasyr
pset2=psett[(psett['activity']=='/v1/activities/2')  ]
pset2
pset1=psett[(psett['activity']=='/v1/activities/1')  ]
pset1


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
    
#check waarde om verschil te begrijpen
pset2r=mkrestcat(pset2,{23:1,54:-0.2,55:-1},26)
sns.scatterplot(data=pd.concat([pset2r,pset2]),x="validfrom",y="volume",hue="energytype")    
# -

pset1r=mkrestcat(pset1,{0:1,1:-1,2:-1,17:-1},10)
#print(pset1r)
#waarde om rest opwek te begrijpen
sns.scatterplot(data=pd.concat([pset1r,pset1]),x="validfrom",y="volume",hue="energytype")  

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
sns.scatterplot(data=pset1tkw,x="validfrom",y="volume",hue="energytype")  
#reconstruct old data subsets
yset2=egasyr[(egasyr['activity']=='/v1/activities/2')  ]
yset2
yset1=egasyr[(egasyr['activity']=='/v1/activities/1')  ]
yset1

yset1t=mkrestcat(egasyr ,{0:1,23:1},10)
yset1tkw=yset1t.copy()
yset1tkw['volume']=yset1tkw['volume']/1000
sns.scatterplot(data=yset1tkw,x="validfrom",y="volume",hue="energytype") 

yset1t0=yset1[yset1['energytypenr']==0]
yset1t0_2023_sum=349280/3.6
yset1t0['volume'].sum()*3.6/1e6
yset1[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6/1e6

yset2t23=yset2[yset2['energytypenr']==23]
yset2t23_2023_sum=800547
yset2[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6/1e6

yset7t23_2023_sum=432838
yset7t23=yset2t23.copy()
yset7t23['volume']= yset7t23_2023_sum*1e6/3.6/(24*365)
yset7t23['energytypenr']= 26
yset7t23['energytype']= "26 - Voertuigbrandstoffen"
yset7t23[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6/1e6

yset7t0=mkrestcat(pd.concat ( [egasyr,yset7t23]) ,{0:1,23:1,26:1},0)
print(yset2t23_2023_sum+ yset7t23_2023_sum+ yset1t0_2023_sum )
yset7t0[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6/1e6

genmult=9
yset8t0=mkrestcat(egasyr ,{1:genmult,2:genmult,17:genmult},1)
yset8t0[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6/1e6

landyrframe = yset8t0[['volume',"validfrom"]].rename(columns={"volume":"opwek"})
landyrframe = landyrframe.merge ( yset7t0[['volume',"validfrom"]].rename(
       columns={"volume":"verbruik"}) ).sort_values("validfrom")
landyrframe.dtypes

landyrframe ["balans"]= landyrframe ["opwek"]- landyrframe ["verbruik"]
landyrframe ["cumbalans"]= landyrframe ["balans"].cumsum()
print(landyrframe ["balans"].sum()*3.6/1e6)
sns.scatterplot(data=landyrframe,x="validfrom",y="cumbalans") 

balansfreq0=landyrframe [["balans"]].sort_values("balans",ascending=False).copy().reset_index()
balansfreq0['n']=balansfreq0.index
balansfreq0['totpwr']=balansfreq0['balans'].cumsum()
sns.scatterplot(data=balansfreq0,x="totpwr",y="balans") 

nhrslong=4*24
longsteff=0.5
longststart=15e9
longstthresh=10e7
#nhrslong=2
landyrframe ["multdaybalans" ]= (landyrframe ["cumbalans" ].shift(-nhrslong)-
                                landyrframe ["cumbalans" ])/nhrslong
landyrframe ["multdaybalanssm" ]= np.convolve(landyrframe ["multdaybalans" ],
                                              np.ones(nhrslong)/nhrslong,mode='same' )
landyrframe ["tolongterm" ] = landyrframe ["multdaybalans" ]. where(
       landyrframe ["multdaybalans" ]<0, landyrframe ["balans" ]. where(
           landyrframe ["balans" ]>longstthresh,0) )
landyrframe ["longtermst" ] = landyrframe ["tolongterm" ] . where(
      landyrframe ["tolongterm" ]<0,landyrframe ["tolongterm" ] *longsteff) .cumsum() +longststart
sns.scatterplot(data=landyrframe,x="validfrom",y="longtermst") 

shortsteff=0.8
shortststart=8e9
landyrframe ["toshortterm" ] = landyrframe ["balans" ] -landyrframe ["tolongterm" ]
landyrframe ["shorttermst" ] = landyrframe ["toshortterm" ] . where(
      landyrframe ["toshortterm" ]<0,landyrframe ["toshortterm" ] *shortsteff) .cumsum() +shortststart
#sns.scatterplot(data=landyrframe,x="validfrom",y="toshortterm") 
sns.scatterplot(data=landyrframe,x="validfrom",y="shorttermst") 


