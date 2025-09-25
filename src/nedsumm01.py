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

energytypes=pd.read_csv("../data/energytypes.txt",sep=" ").set_index('index')
energytypes
energytypes_dict=energytypes.to_dict()['Energytype']
energytypes_dict

#haal gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
egasyr=pd.read_pickle("../intermediate/egasyr2024.pkl")
#egasyr.dtypes
#en doe berekeningen in GWh, en niet in kWh
egasyr['volume']=egasyr['volume']*1e-6
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
pset1tkw['volume']=pset1tkw['volume']
sns.scatterplot(data=pset1tkw,x="validfrom",y="volume",hue="energytype")  
#reconstruct old data subsets
yset2=egasyr[(egasyr['activity']=='/v1/activities/2')  ]
yset2
yset1=egasyr[(egasyr['activity']=='/v1/activities/1')  ]
yset1

yset1t=mkrestcat(egasyr ,{0:1,23:1},10)
yset1tkw=yset1t.copy()
yset1tkw['volume']=yset1tkw['volume']
sns.scatterplot(data=yset1tkw,x="validfrom",y="volume",hue="energytype") 

yset1t0=yset1[yset1['energytypenr']==0]
yset1t0_2023_sum=349280
print(yset1t0['volume'].sum()*3.6)
yset1[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6

yset2t23=yset2[yset2['energytypenr']==23]
yset2t23_2023_sum=800547
print(yset2t23['volume'].sum()*3.6)
yset2[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6

#voeg voertuigbrandstoffen toe, in gelijke mate per uur
#daardoor ontstaat niet-gepiekt (wel planbaar !) verbruik
yset7t23_2023_sum=432838
yset7t23=yset2t23.copy()
yset7t23['volume']= yset7t23_2023_sum/3.6/(24*365)
yset7t23['energytypenr']= 26
yset7t23['energytype']= "26 - Voertuigbrandstoffen"
yset7t23[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6

yset7t0=mkrestcat(pd.concat ( [egasyr,yset7t23]) ,{0:1,23:1,26:1},0)
print(yset2t23_2023_sum+ yset7t23_2023_sum+ yset1t0_2023_sum )
yset7t0[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6

genmult=9
yset8t0=mkrestcat(egasyr ,{1:genmult,2:genmult,17:genmult},1)
yset8t0[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6

landyrframe = yset8t0[['volume',"validfrom"]].rename(columns={"volume":"opwek"})
landyrframe = landyrframe.merge ( yset7t0[['volume',"validfrom"]].rename(
       columns={"volume":"verbruik"}) ).sort_values("validfrom")
landyrframe.dtypes

#cumulatieve balans over het jaar, in GWh
landyrframe ["balans"]= landyrframe ["opwek"]- landyrframe ["verbruik"]
landyrframe ["cumbalans"]= landyrframe ["balans"].cumsum()
print(landyrframe ["balans"].sum()*3.6)
sns.lineplot(data=landyrframe,x="validfrom",y="balans") 
plt.title("uurbalansen: positief is overschot opwek")

sns.lineplot(data=landyrframe,x="validfrom",y="cumbalans") 
plt.title("cumulatieve uurbalansen")

#extra opwek (nodig vanwege opslag verliezen) relatief t.o.v. jaarverbruik
landyrframe ["cumbalansrel"]= landyrframe ["cumbalans"]/ (landyrframe ["verbruik"].sum())
sns.lineplot(data=landyrframe,x="validfrom",y="cumbalansrel") 
plt.title("cumulatieve uurbalansen als fractie van jaarverbruik")


# +
#grafiek: te lezen vanaf links
#als je opslag laadt vanaf een bepaalde grens (GW , of GWh/uur), hoe veel 
#totaal niet direct verbruikt vermogen (GWh) is dan per jaar beschikbaar ?
def balansstats(df,col,totpwr):
    balansfreq0=landyrframe [[col]].sort_values(col,ascending=False).copy().reset_index()
    balansfreq0['n']=balansfreq0.index
    balansfreq0['totpwr']=balansfreq0[col].cumsum()
    hrload= (balansfreq0[col] >0).sum()
    hrdis= (balansfreq0[col] <0).sum()
    tit=(" long term uren laden: %d (%d %%), uren ontladen %d (%d %%)"%(hrload,hrload*(100/(24*365)),hrdis,hrdis*(100/(24*365))))
    if totpwr:
        sns.lineplot(data=balansfreq0,y="totpwr",x="balans") 
        plt.xlabel("bij overschotten boven dit vermogen (GW)")
        plt.ylabel("blijft jaarlijks dit over (GWh)")
    else:
        p=sns.lineplot(data=balansfreq0,x="n",y=col) 
        plt.title(tit)

balansstats(landyrframe, "balans",True)
# -

#grafiek: te lezen vanaf links
#als je opslag laadt vanaf een bepaalde grens (GW , of GWh/uur), hoe veel 
#totaal niet direct verbruikt vermogen (GWh) is dan per jaar beschikbaar ?
balansfreq0=landyrframe [["balans"]].sort_values("balans",ascending=False).copy().reset_index()
balansfreq0['n']=balansfreq0.index
balansfreq0['totpwr']=balansfreq0['balans'].cumsum()
sns.lineplot(data=balansfreq0,y="totpwr",x="balans") 

#eerste beschrijving van long-term storage: opgeslagen vermogen in GWh
#gaat uit van snel laden, grote verliezen bij laden en weinig verlies over tijd
#grafiek mag niet onder 0 uit komen, en moet aan einde royaal hoger dan begin uit komen
nhrslong=4*24
longsteff=0.4
longststart=15e3
longstthresh=80
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
sns.lineplot(data=landyrframe,x="validfrom",y="longtermst") 


# +
def balansstats(df,col):
    balansfreq0=landyrframe [[col]].sort_values(col,ascending=False).copy().reset_index()
    balansfreq0['n']=balansfreq0.index
    balansfreq0['totpwr']=balansfreq0[col].cumsum()
    hrload= (balansfreq0[col] >0).sum()
    hrdis= (balansfreq0[col] <0).sum()
    tit=(" long term uren laden: %d (%d %%), uren ontladen %d (%d %%)"%(hrload,hrload*(100/(24*365)),hrdis,hrdis*(100/(24*365))))
    p=sns.lineplot(data=balansfreq0,x="n",y=col) 
    plt.title(tit)

balansstats(landyrframe, "tolongterm")
# -

#dan moet de rest opgevangen door korte termijn
shortsteff=0.8
shortststart=500
landyrframe ["toshortterm" ] = landyrframe ["balans" ] -landyrframe ["tolongterm" ]
landyrframe ["shorttermst" ] = landyrframe ["toshortterm" ] . where(
      landyrframe ["toshortterm" ]<0,landyrframe ["toshortterm" ] *shortsteff) .cumsum() +shortststart
sns.lineplot(data=landyrframe,x="validfrom",y="toshortterm") 
#sns.lineplot(data=landyrframe,x="validfrom",y="shorttermst") 

sns.lineplot(data=landyrframe,x="validfrom",y="shorttermst") 

balansstats(landyrframe, "toshortterm")


