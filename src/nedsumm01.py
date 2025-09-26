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
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import subprocess
import requests
import json
import os

# +
#nu opslag parametriseren, eerst generiek model
# -

feature_list=['tosto']
stoex = pd.DataFrame(0.0, index=np.arange(100), columns=feature_list) 
stoex.at[3,"tosto"]=1
stoex.plot()
stoex.dtypes


def memfunc(times,initst,halfw,mins,maxs):
    state=initst
    decval=1-np.log(2)/halfw
#    print(decval)
    a=times.copy()
    with np.nditer(a, op_flags=['readwrite'], order='K') as it:
        for x in it:
            state += x
            if state > maxs:
                state=maxs
            if state < mins:
                state=mins
            x[...] =  state
            state *=decval
    return a
stoex['storst1']=memfunc(stoex['tosto'],0.0,40,0,0.5)
print(stoex['storst1'][43])
stoex['storst1'].plot()

# +
#ophalen generieke gegevens
# -

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
                          "energytype","energytypenr","emissionfactor","lastupdate",
                          "activity"])
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

gr3usdf=pd.concat ( [egasyr,yset7t23]) 
gr3usdf=gr3usdf[gr3usdf['energytypenr'].isin ({0,23,26})].copy()
gr3usdf['energytype']=gr3usdf['energytype'].where(gr3usdf['energytypenr']!=0,'0 - Elektriciteit')
sns_plot=sns.lineplot(data=gr3usdf,x="validfrom",y="volume",hue='energytype') 
plt.title("uurwaarden verbruik ")
plt.ylabel("Uurvermogen (GW) of (Gwh/hr)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
figname = "../output/gebrtot3gr2024"+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

gr3usdfc=gr3usdf.copy(deep=False)
gr3usdfc['volcum']=gr3usdfc.groupby(['energytype'])['volume'].cumsum()
totverbruik = gr3usdfc['volume'].sum()
title=  ("Opbouw jaarverbruik 3 groepen\n totaal %.0f GWh /jaar, %.0f GWh /dag" %(
    totverbruik,totverbruik/365))
sns_plot= sns.lineplot(data=gr3usdfc,x="validfrom",y="volcum",hue='energytype') 
plt.title(title)
plt.ylabel("Cumulatief verbruik (Gwh)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
figname = "../output/gebrtot3grc2024"+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

yset7t0=mkrestcat(gr3usdf,{0:1,23:1,26:1},0)
print(yset2t23_2023_sum+ yset7t23_2023_sum+ yset1t0_2023_sum )
yset7t0[['volume','energytype']].groupby('energytype').agg('sum')*3.6

# +
#nu zelfde excercitie voor opwek
# -

gr3usdf=egasyr
gr3usdf=gr3usdf[gr3usdf['energytypenr'].isin ({1,2,17})].copy()
gr3usdf['energytype']=gr3usdf['energytype'].where(gr3usdf['energytypenr']!=0,'0 - Elektriciteit')
sns_plot=sns.scatterplot(data=gr3usdf,x="validfrom",y="volume",hue='energytype') 
plt.title("uurwaarden opwek ")
plt.ylabel("Uurvermogen (GW) of (GWh/hr)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
figname = "../output/opwtot3gr2024"+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

gr3usdfc=gr3usdf.copy(deep=False)
gr3usdfc['volcum']=gr3usdfc.groupby(['energytype'])['volume'].cumsum()
totopwek = gr3usdfc['volume'].sum()
title=  ("Opbouw jaaropwek 3 groepen\n totaal %.0f GWh /jaar, %.0f GWh /dag" %(
    totopwek,totopwek/365))
sns_plot= sns.lineplot(data=gr3usdfc,x="validfrom",y="volcum",hue='energytype') 
plt.title(title)
plt.ylabel("Cumulatieve opwek (GWh)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
figname = "../output/opwtot3grc2024"+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

# +
#inst_opw bepaalt opwek instelling
inst_opw='A'

some_string="""inst,windmult,zonrel
A,9,1
B,12,0.5"""
#read CSV string into pandas DataFrame
param_opw= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst').to_dict('index')[inst_opw]
print(param_opw)

yset8t0=mkrestcat(egasyr ,{1:param_opw['windmult'],
                           2:param_opw['windmult'] * param_opw['zonrel'],
                           17:param_opw['windmult']},1)
yset8t0[['volume','energytype']].groupby('energytype').agg('sum')*3.6
# -

landyrframe = yset8t0[['volume',"validfrom"]].rename(columns={"volume":"opwek"})
landyrframe = landyrframe.merge ( yset7t0[['volume',"validfrom"]].rename(
       columns={"volume":"verbruik"}) ).sort_values("validfrom").copy()
landyrframe.dtypes

sns.lineplot(data=landyrframe,x="validfrom",y="opwek") 
sns_plot=sns.lineplot(data=landyrframe,x="validfrom",y="verbruik") 
dagverbruik=landyrframe['verbruik'].sum()/365
print (dagverbruik)
plt.title("uurwaarden verbruik 2024 (max= %.0f) en opwek x9 (max= %.0f)"% (
   landyrframe["verbruik"].max(),landyrframe["opwek"].max()))
plt.ylabel("Uurvermogen (GW) of (Gwh/hr)")
figname = "../output/eneuthr2024hr_"+inst_opw+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

#cumulatieve balans over het jaar, in GWh
landyrframe ["balans"]= landyrframe ["opwek"]- landyrframe ["verbruik"]
landyrframe ["cumbalans"]= landyrframe ["balans"].cumsum()
print(landyrframe ["balans"].sum()*3.6)
sns.lineplot(data=landyrframe,x="validfrom",y="balans") 
plt.title("uurbalansen: positief is overschot opwek")

sns_plot=sns.lineplot(data=landyrframe,x="validfrom",y="cumbalans") 
lastw=landyrframe.tail(1)["cumbalans"]
totv=(landyrframe ["verbruik"].sum())
title= "cumulatieve uurbalansen, overschot = %.0f (= %.1f %% van jaarverbruik)" % (
     lastw,100*lastw/totv)
plt.title(title)
figname = "../output/eneuthr2024c_"+inst_opw+'.png';
sns_plot.figure.savefig(figname,dpi=300) 

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

#keuzes modellen
glb_inst_long='A'
glb_inst_short='A'
inst_str=inst_opw+glb_inst_long+glb_inst_short

# +
#eerste beschrijving van long-term storage: opgeslagen vermogen in GWh
#gaat uit van snel laden, grote verliezen bij laden en weinig verlies over tijd
#grafiek mag niet onder 0 uit komen, en moet aan einde royaal hoger dan begin uit komen
#bij een longstthresh hoger dan 80 raakt de short term overvol in zomer
some_string="""inst,ndayslong,steff,ststart,stthresh,tfact,athresh,afact,ofact,yrhalf
A,4,0.4,15e3,80,1,0,0,1.1,2
B,4,0.4,15e3,0,0,20,0.9,1.1,2"""
    #read CSV string into pandas DataFrame    
param_longdf= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')

def add_longst_io(df,my_inst_long,my_inst_str):
    param_long=param_longdf.to_dict('index')[my_inst_long]
    print(param_long)

    nhrslong=24*param_long['ndayslong']
    #longsteff=param_long['steff']
    #nhrslong=2
    
    repyr=pd.concat ( [ df ["balans" ], df ["balans" ][0:nhrslong]  ] ).cumsum()
    bcalc= ( repyr.shift(-nhrslong)- repyr)/nhrslong
    df ["multdaybalans" ]= bcalc[0:len(df.index)]
    df ["multdaybalanssm" ]= np.convolve(df ["multdaybalans" ],
                                             np.ones(nhrslong)/nhrslong,mode='same' )
    df ["tolongterm" ] = ( df ["multdaybalans" ] *  param_long['ofact'] ) . where(
           df ["multdaybalans" ]<0,
           param_long['afact'] *df ["multdaybalans" ]* (df ["multdaybalans" ] >param_long['athresh'] )) + ( 
            ( param_long['tfact'] * df ["balans" ] ) . where(
               df ["balans" ]>param_long['stthresh'],0) )
    df ["longtermst" ] = df ["tolongterm" ] . where(
          df ["tolongterm" ]<0,df ["tolongterm" ] *param_long['steff']
           ) .cumsum() +param_long['ststart']
    sns_plot=sns.scatterplot(data=df,x="validfrom",y="tolongterm") 
    title= 'Long-time storage usage: out smooth %.0f days, in if hour > %0d GW'%(
        param_long['ndayslong'],param_long['stthresh'])
    plt.title(title)
    figname = "../output/longst_io_"+inst_str+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
add_longst_io(landyrframe ,glb_inst_long,inst_str)    
# -

sns.lineplot(data=landyrframe,x="validfrom",y="longtermst") 


#longststart
def add_longst_mem(df,my_inst_long,my_inst_str):
    param_long=param_longdf.to_dict('index')[my_inst_long]
    df ["longtermsd" ] = memfunc(df ["tolongterm" ] . where(
          df ["tolongterm" ]<0,df ["tolongterm" ] *param_long['steff']),
          param_long['ststart'],param_long['yrhalf']*365*24,0,4*param_long['ststart'])  
    empty=df [df ["longtermsd" ] ==0 ].copy().reset_index()
    if empty.size !=0:
        print("WARNING: storage gets empty")
        print(empty["validfrom"])
    else:    
        print("OK: storage does not get empty")
    if (df.tail(1)["longtermsd"] < param_long['ststart']).any()  :
        print("WARNING: storage depleted over year %.0f < %.0f" % (df.tail(1)["longtermsd"] , param_long['ststart']))
    else:        
        print("OK: storage surplus over year %.0f >%.0f" % (df.tail(1)["longtermsd"] , param_long['ststart']))
    stomax=df["longtermsd"].max()
    sns.lineplot(data=df,x="validfrom",y="longtermst",label="cum. balans") 
    sns_plot=sns.lineplot(data=df,x="validfrom",y="longtermsd",label="storage") 
    title= 'Long-time storage filling\n'
    title = title +('storage cycle eff %.0f %% ,half-time %.1f yr, max = %.0f initial = %.0f'% (
        param_long['steff']*100,param_long['yrhalf'],stomax,param_long['ststart']))
    plt.title(title)
    figname = "../output/longst_fill_"+my_inst_str+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
add_longst_mem(landyrframe ,glb_inst_long,inst_str)        


# +
def balansstats(df,col):
    balansfreq0=landyrframe [[col]].sort_values(col,ascending=False).copy().reset_index()
    balansfreq0['n']=balansfreq0.index
    balansfreq0['totpwr']=balansfreq0[col].cumsum()
    hrload= (balansfreq0[col] >0).sum()
    hrdis= (balansfreq0[col] <0).sum()
    tit=("%s uren laden: %d (%d %%), uren ontladen %d (%d %%)"%(col,hrload,hrload*(100/(24*365)),hrdis,hrdis*(100/(24*365))))
    sns_plot=sns.lineplot(data=balansfreq0,x="n",y=col) 
    plt.title(tit)
    figname = "../output/"+col+"iohrs_"+inst_str+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 

balansstats(landyrframe, "tolongterm")

# +
some_string="""inst,steff,ststart,wkhalf,dayscap
A,0.9,4000,4,7
B,0.9,4000,4,7"""
    #read CSV string into pandas DataFrame
param_shortdf= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')

#dan moet de rest opgevangen door korte termijn
def add_shortst_io(df,my_inst_long,my_inst_short,my_inst_str):
    param_short= param_shortdf.to_dict('index')[my_inst_short]
    print(param_short)

    shortsteff=param_short['steff']
    shortststart=param_short['ststart']
    df ["toshortterm" ] = df ["balans" ] -df ["tolongterm" ]
    df ["shorttermst" ] = df ["toshortterm" ] . where(
          df ["toshortterm" ]<0,df ["toshortterm" ] *shortsteff) .cumsum() +shortststart
    sns_plot=sns.lineplot(data=df,x="validfrom",y="toshortterm") 
    #sns.lineplot(data=df,x="validfrom",y="shorttermst") 
    param_long=param_longdf.to_dict('index')[my_inst_long]
    title= 'Short-time storage usage\nLong out smooth %.0f days, in if hour > %0d GW'%(
          param_long['ndayslong'],param_long['stthresh'])
    plt.title(title)
    figname = "../output/shortst_io_"+inst_str+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
add_shortst_io(landyrframe ,glb_inst_long,glb_inst_short,inst_str)    
# -

#opslag zonder halfwaardetijd en maxima
sns.lineplot(data=landyrframe,x="validfrom",y="shorttermst") 

balansstats(landyrframe, "toshortterm")


# +
#nu opslag model toepassen
# -

def add_shortst_mem(df,my_inst_short,my_inst_str):
    param_short= param_shortdf.to_dict('index')[my_inst_short]
    shortsteff=param_short['steff']
    shortststart=param_short['ststart']

    shortmaxsto=totverbruik*param_short['dayscap']/365
    shorthalfw=param_short['wkhalf']*7*24
    df ["shorttermsd" ] = memfunc(df ["toshortterm" ] . where(
          df ["toshortterm" ]<0,df ["toshortterm" ] *shortsteff) 
                                           ,shortststart,shorthalfw,0,shortmaxsto)  
    empty=df [df ["shorttermsd" ] ==0 ].copy().reset_index()
    if empty.size !=0:
        print("WARNING: storage gets empty")
        print(empty["validfrom"])
    else:    
        print("OK: storage does not get empty")
    if (df.tail(1)["shorttermsd"] < shortststart).any()  :
        print("WARNING: storage depleted over year %.0f < %.0f" % (df.tail(1)["shorttermsd"] , shortststart))        

    sns_plot=sns.lineplot(data=df,x="validfrom",y="shorttermsd") 
    stomax=df["shorttermsd"].max()
    title= 'Short-time storage filling\n'
    title = title +('storage cycle eff %.0f %% ,half-time %.0f days, max = %.0f initial = %.0f'% (
        shortsteff*100,shorthalfw/24,stomax,shortststart))
    plt.title(title)
    figname = "../output/shortst_fill_"+my_inst_str+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
add_shortst_mem(landyrframe ,glb_inst_short,inst_str)        

#nu run opnieuw met andere parameters
glb_inst_long='B'
glb_inst_short='A'
inst_str=inst_opw+glb_inst_long+glb_inst_short
add_longst_io(landyrframe ,glb_inst_long,inst_str)    

balansstats(landyrframe, "tolongterm")

add_longst_mem(landyrframe ,glb_inst_long,inst_str)   

add_shortst_io(landyrframe ,glb_inst_long,glb_inst_short,inst_str)        

balansstats(landyrframe, "toshortterm")

add_shortst_mem(landyrframe ,glb_inst_short,inst_str) 


