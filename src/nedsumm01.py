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

from ipywidgets import Dropdown, Button, Output, interact
from IPython.display import display

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

#referentie gegevens in TJ
regklimmon23_map={0: 349280, 23: 800547, 26: 432838}

# +
#ophalen jaargegevens
# -

yrtomodel='2024'


#haal gegevens op, zodat laden (waar API key voor nodig is) maak 1 maal hoeft
def getyrdta(my_yrtomodel):
    egasyr=pd.read_pickle("../intermediate/egasyr"+yrtomodel+".pkl")
    #egasyr.dtypes
    #en doe berekeningen in GWh, en niet in kWh
    egasyr['volume']=egasyr['volume']*1e-6
    return( egasyr)
egasyr= getyrdta(yrtomodel)    
egasyr

#maak kleine sets om de eerste paar dagen mee te plotten
maxd=yrtomodel+"-01-16T00:00:00+00:00"
psett=egasyr[(egasyr['validfrom']< pd.to_datetime(maxd))]
#maar gebruik nu hele jaar
psett=egasyr
pset2=psett[(psett['activity']=='/v1/activities/2')  ]
pset2
pset1=psett[(psett['activity']=='/v1/activities/1')  ]
pset1

#hieruit kan e.v.t worden geprobeerd de 25 % missend gasverbruik te reconstrueren
gassums= pset2.groupby(["energytype","energytypenr"])['volume'].agg('sum')*3.6/ regklimmon23_map[23]
print (gassums)
dv=gassums.reset_index('energytype')['volume'].to_dict()
print(dv[23]+dv[31])
print(dv[53]+dv[54]+dv[55])


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
    
plt.clf()
#check waarde om verschil te begrijpen
pset2r=mkrestcat(pset2,{23:1,54:-0.2,55:-1},26)
#print(pset2.dtypes)
#print(pset2.agg('min'))
sns.scatterplot(data=pd.concat([pset2r,pset2]),x="validfrom",y="volume",hue="energytype")    
# -

pset1r=mkrestcat(pset1,{0:1,1:-1,2:-1,17:-1},10)
#print(pset1r)
#waarde om rest opwek te begrijpen
sns.scatterplot(data=pd.concat([pset1r,pset1]),x="validfrom",y="volume",hue="energytype")  

#maak plot eerste dagen
pset1t=mkrestcat(pd.concat ( [pset1,pset2]) ,{0:1,23:1},10)
pset1tkw=pset1t.copy()
pset1tkw['volume']=pset1tkw['volume']
sns.scatterplot(data=pset1tkw,x="validfrom",y="volume",hue="energytype")  

some_string="""inst,vbfact,vbverd,omschr
A,1,0,data + voertuig glad"""
#read CSV string into pandas DataFrame
param_verbr_df= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')
print(param_verbr_df)


# +
def mkcombiset(yrindat):
    yset2=yrindat[(yrindat['activity']=='/v1/activities/2')  ]
#    yset2
    yset1=yrindat[(yrindat['activity']=='/v1/activities/1')  ]
#    yset1
    

    yset1t=mkrestcat(yrindat ,{0:1,23:1},10)
    yset1tkw=yset1t.copy()
    yset1tkw['volume']=yset1tkw['volume']
    sns.scatterplot(data=yset1tkw,x="validfrom",y="volume",hue="energytype") 
    yset1t0=yset1[yset1['energytypenr']==0]

    yset2t23=yset2[yset2['energytypenr']==23]
    #voeg voertuigbrandstoffen toe, in gelijke mate per uur
    #daardoor ontstaat niet-gepiekt (wel planbaar !) verbruik
    yset7t23_2023_sum=regklimmon23_map[26]
    yset7t23=yset2t23.copy()
    yset7t23['volume']= yset7t23_2023_sum/3.6/(24*365)
    yset7t23['energytypenr']= 26
    yset7t23['energytype']= "26 - Voertuigbrandstoffen"
    gr3iusdf= pd.concat ( [yrindat,yset7t23]) 
    gr3iusdf=gr3iusdf[gr3iusdf['energytypenr'].isin ({0,23,26})].copy()
    gr3iusdf['energytype']=gr3iusdf['energytype'].where(gr3iusdf['energytypenr']!=0,'0 - Elektriciteit')
    v3=(gr3iusdf[['volume','energytypenr']].groupby('energytypenr').agg('sum')*3.6 ).reset_index()
    v3['regklimmon23_TJ'] = v3['energytypenr'].map(regklimmon23_map)
    v3['regklimmon23rat'] = v3['volume']/ v3['regklimmon23_TJ'] 
    print(v3)
    return (gr3iusdf)

gr3usdf=mkcombiset(egasyr)


# -
def mkuurpl3gr(df,gtype,grprfx,my_yrtomodel,labpl):
    sns_plot=sns.lineplot(data=df,x="validfrom",y="volume",hue='energytype',ci=None) 
    plt.title("uurwaarden "+gtype+my_yrtomodel)
    plt.ylabel("Uurvermogen (GW) of (Gwh/hr)")
    plt.legend(bbox_to_anchor=(labpl, 0.99), loc=0, borderaxespad=0.)
    figname = "../output/"+grprfx+"tot3gr"+my_yrtomodel+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
mkuurpl3gr(gr3usdf,"verbruik ",'gebr',yrtomodel,0.3)


def mkuurpl3grc(df,gtype,grprfx,my_yrtomodel):
    gr3usdfc=df.copy(deep=False)
    gr3usdfc['volcum']=gr3usdfc.groupby(['energytype'])['volume'].cumsum()
    totverbruik = gr3usdfc['volume'].sum()
    title=  ("Opbouw %s%s: 3 groepen\n totaal %.0f GWh /jaar, %.0f GWh /dag" %(
        gtype,my_yrtomodel,totverbruik,totverbruik/365))
    sns_plot= sns.lineplot(data=gr3usdfc,x="validfrom",y="volcum",hue='energytype',ci=None) 
    plt.title(title)
    plt.ylabel("Cumulatief verbruik (Gwh)")
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.)
    figname = "../output/"+grprfx+"tot3grc"+my_yrtomodel+'.png';
    sns_plot.figure.savefig(figname,dpi=300) 
mkuurpl3grc(gr3usdf,"verbruik ",'gebr',yrtomodel)    

#bereken totaal verbruik, voor verder model
yset7t0=mkrestcat(gr3usdf,{0:1,23:1,26:1},0)
#print(yset2t23_2023_sum+ yset7t23_2023_sum+ yset1t0_2023_sum )
yset7t0[['volume','energytype']].groupby('energytype').agg('sum')*3.6

# +
#nu zelfde excercitie voor opwek
# -

mkuurpl3gr(egasyr[egasyr['energytypenr'].isin ({1,2,17})].copy(),"opwek ",'opw',yrtomodel,0.8)

mkuurpl3grc(egasyr[egasyr['energytypenr'].isin ({1,2,17})].copy(),"opwek ",'opw',yrtomodel)

figcmb, axcmb = plt.subplots(nrows=3, ncols=3,figsize=(10, 8))
figcmb.tight_layout(pad=4)

# +
#inst_opw bepaalt opwek instelling
glb_inst_opw='A'
inst_str=yrtomodel+glb_inst_opw

some_string="""inst,windmult,zonrel,omschr
A,9.0,1,evenredig zon-wind
B,12,0.5,meer wind
C,6,2.5,meer zon
D,7,1,energieneutraal+6%"""
#read CSV string into pandas DataFrame
param_opw_df= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')
print(param_opw_df)

def get_param_opw(yrstr,inst_opw):
    param_opw = param_opw_df.to_dict('index')[inst_opw]
    ytmult=1
    if yrstr=='2023':
        ytmult=1.1
    param_opw['windmult']= (param_opw['windmult']*ytmult)
    return(param_opw)

def mkusopw(egasindf,yset7t0df,yrstr,inst_opw):
    param_opw=get_param_opw(yrstr,inst_opw)
    print(param_opw)

    yset8t0=mkrestcat(egasindf,{1:param_opw['windmult'],
                               2:param_opw['windmult'] * param_opw['zonrel'],
                               17:param_opw['windmult']},1)
#    totvol=yset8t0[['volume','energytype']].groupby('energytype').agg('sum')*3.6
#    print(totvol)
    oframe= yset8t0[['volume',"validfrom"]].rename(columns={"volume":"opwek"})
    voorwarmte = egasindf[egasindf['energytypenr'].isin ({23})] . rename(columns={"volume":"warmtevbr"})
    voorwarmte ["warmtevbr"]=  voorwarmte ["warmtevbr"] - (0.5*voorwarmte ["warmtevbr"].min())                      
    tframe = oframe.merge ( voorwarmte[['warmtevbr',"validfrom"]])
    cframe = tframe.merge ( yset7t0df[['volume',"validfrom"]].rename(
           columns={"volume":"verbruik"}) ).sort_values("validfrom").copy()    
    cframe ["balans"]= cframe ["opwek"]- cframe ["verbruik"]
    cframe ["cumbalans"]= cframe ["balans"].cumsum()
    totvol=cframe[['verbruik','opwek','balans','warmtevbr']].agg('sum')*3.6
    print(totvol)
    return (cframe)
               
landyrframe= mkusopw(egasyr,yset7t0,yrtomodel,glb_inst_opw)
landyrframe.dtypes            


# -

def mkovplot(dfin,yrstr,my_inst_opw,my_inst_str,ax):
    param_opw=get_param_opw(yrstr,my_inst_opw)
#    print(param_opw)
    df=dfin.copy(deep=False)
    sns.lineplot(ax=ax,data=df,x="validfrom",y="opwek",ci=None,label='opwek') 
    sns_plot=sns.lineplot(ax=ax,data=df,x="validfrom",y="verbruik",ci=None,label='verbruik')  
    avgverbruik=df['verbruik'].mean()    
    ptit=("Verbruik %s (avg = %0.f max= %.0f) en wind * %.1f + zon * %.1f (max= %.0f)"% (
       inst_str,avgverbruik,df["verbruik"].max(),
       param_opw['windmult'] ,param_opw['windmult'] * param_opw['zonrel'] , df["opwek"].max()))
    df['gelijktijdig'] = df['verbruik'].where(df['verbruik'] < df['opwek'],df['opwek'] )
#    sns.lineplot(data=df,x="validfrom",y="gelijktijdig",ci=None) 
    opwekrat=df['opwek'].mean() /avgverbruik    
    gelijktrat=df['gelijktijdig'].mean() /avgverbruik
    ptit=ptit+("\ngem dagverbr= %.0f GWh, opwek %0.f %% verbruik, gelijktijdig %0.f %% verbruik %.0f %% opwek)"% (
       avgverbruik*24, opwekrat*100,gelijktrat*100, gelijktrat*100/opwekrat) )
    ax.set_title(ptit)
    ax.legend(bbox_to_anchor=(0.75, 0.99), loc=0, borderaxespad=0.)
    ax.set_ylabel("Uurvermogen (GW) of (Gwh/hr)")
    ax.set_xlabel("datum (gegevens per uur)")
#    figname = "../output/eneuthr_hr_"+my_inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
mkovplot(landyrframe,yrtomodel,glb_inst_opw,inst_str,axcmb[0,0])    

# balans over het jaar, in GWh
sns.lineplot(data=landyrframe,x="validfrom",y="balans",ci=None) 
plt.title("uurbalansen: positief is overschot opwek")


def cumbalplot(cframe,my_inst_str,ax):
    sns_plot=sns.lineplot(ax=ax,data=cframe,x="validfrom",y="cumbalans",ci=None) 
    lastw=cframe.tail(1)["cumbalans"]
    totv=(cframe ["verbruik"].sum())
    title= "cumulatieve uurbalansen %s, overschot = %.0f (= %.1f %% van jaarverbruik)" % (
         inst_str,lastw,100*lastw/totv)
    ax.set_title(title)
#    figname = "../output/eneuthr_cum_"+inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
cumbalplot(landyrframe,inst_str,axcmb[0,1])    


#extra opwek (nodig vanwege opslag verliezen) relatief t.o.v. jaarverbruik
def plotreovsch(cframe):
    cframe ["cumbalansrel"]= cframe["cumbalans"]/ (cframe ["verbruik"].sum())
    sns.lineplot(data=cframe,x="validfrom",y="cumbalansrel",ci=None) 
    plt.title("cumulatieve uurbalansen als fractie van jaarverbruik")
plotreovsch(landyrframe.copy(deep=False))    


# +
#grafiek: te lezen vanaf links
#als je opslag laadt vanaf een bepaalde grens (GW , of GWh/uur), hoe veel 
#totaal niet direct verbruikt vermogen (GWh) is dan per jaar beschikbaar ?
def balansstats(df,col,totpwr,my_inst_str,ax):
    balansfreq0=df[[col]].sort_values(col,ascending=False).copy().reset_index()
    balansfreq0['n']=balansfreq0.index
    balansfreq0['totpwr']=balansfreq0[col].cumsum()
    hrload= (balansfreq0[col] >0).sum()
    hrdis= (balansfreq0[col] <0).sum()
    if my_inst_str == 'balans':
        tit=(" long term uren laden: %d (%d %%), uren ontladen %d (%d %%)"%(hrload,hrload*(100/(24*365)),hrdis,hrdis*(100/(24*365))))
    else:
        avg= balansfreq0[col].mean()
        hrlow= (balansfreq0[col] < avg/10).sum()
        tit=(" %s %s average %.1f, hrs below %.2f : %.0f"%(col,inst_str,avg,avg/10,hrlow))
#    plt.clf()
    if totpwr:
        p=sns.lineplot(ax=ax,data=balansfreq0,y="totpwr",x="balans",ci=None) 
        ax.set_xlabel("bij overschotten boven dit vermogen (GW)")
        ax.set_ylabel("blijft jaarlijks dit over (GWh)")
#        figname = "../output/"+col+"iocum_"+my_inst_str+'.png';
#        p.figure.savefig(figname,dpi=300) 
    else:
        p=sns.lineplot(ax=ax,data=balansfreq0,x="n",y=col,ci=None) 
        ax.set_title(tit)
#        figname = "../output/"+col+"iohrs_"+my_inst_str+'.png';
#        p.figure.savefig(figname,dpi=300) 

balansstats(landyrframe, "balans",True,inst_str,axcmb[0,2])
# -

#grafiek: te lezen vanaf links
#als je opslag laadt vanaf een bepaalde grens (GW , of GWh/uur), hoe veel 
#totaal niet direct verbruikt vermogen (GWh) is dan per jaar beschikbaar ?
balansfreq0=landyrframe [["balans"]].sort_values("balans",ascending=False).copy().reset_index()
balansfreq0['n']=balansfreq0.index
balansfreq0['totpwr']=balansfreq0['balans'].cumsum()
sns.lineplot(data=balansfreq0,y="totpwr",x="balans",ci=None) 

fig,ax= plt.subplots(figsize=(10,6))
balansstats(landyrframe, "opwek",False,inst_str,ax)

#keuzes modellen
glb_inst_long='A'
glb_inst_short='A'
inst_str=yrtomodel+glb_inst_opw+glb_inst_long+glb_inst_short

# +
#eerste beschrijving van long-term storage: opgeslagen vermogen in GWh
#gaat uit van snel laden, grote verliezen bij laden en weinig verlies over tijd
#grafiek mag niet onder 0 uit komen, en moet aan einde royaal hoger dan begin uit komen
#bij een longstthresh hoger dan 80 raakt de short term overvol in zomer
some_string="""inst,ndayslong,steff,ststart,stthresh,tfact,athresh,afact,ofact,yrhalf,omschr
A,4,0.4,30e3,0,0,15,0.9,1.01,2,4 dgn geleidelijke opslag
B,4,0.4,30e3,70,1,0,0,1.01,2,4 dgn alleen piek opslag
C,2,0.4,30e3,0,0,3,0.95,1.01,2,2 dgn geleidelijke opslag
D,7,0.4,30e3,0,0,15,0.9,1.01,2,7 dgn geleidelijke opslag
E,2,3.0,60e3,0,0,15,0.9,1.01,2,2 dgn warmte eff"""
    #read CSV string into pandas DataFrame    
param_longdf= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')

some_string="""inst,steff,ststart,wkhalf,dayscap,omschr
A,0.9,4000,4,7,4 dgn cap
B,0.9,4000,4,7,kopie van A"""
    #read CSV string into pandas DataFrame
param_shortdf= pd.read_csv(io.StringIO(some_string), sep=",").set_index('inst')


# -

def add_longst_io(df,my_inst_long,my_inst_short,my_inst_str,ax,color):
    param_long=param_longdf.to_dict('index')[my_inst_long]
    print(param_long)

    nhrslong=24*param_long['ndayslong']
    #longsteff=param_long['steff']    
    #bereken de outflow per uur om het short-term storage te onderhouden
    #dit is een gemiddeld systeem verlies dat het short-term storage ieder uur gemiddeld verbruikt
    param_short= param_shortdf.to_dict('index')[my_inst_short]
    shortsteff=param_short['steff']
    shortststart=param_short['ststart']
    shorthalfw=param_short['wkhalf']*7*24
    #bereken gemiddelde verliezen korte termijn
    avgshusg = shortststart * (np.log(2)/shorthalfw) / shortsteff
    #voeg gemiddelde verliezen korte termijn toe aan balans
    balans1= df ["balans" ] - avgshusg
    repyr=pd.concat ( [ balans1, balans1[0:nhrslong]  ] ).cumsum()
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
#    plt.clf()
    sns_plot=sns.scatterplot(ax=ax,data=df,x="validfrom",y="tolongterm",color=color) 
    title= 'Long-time storage in/out %s: out smooth %.0f days, need * %.2f\n in if hour > %0d GW  %.2f + smooth in > %.0d GW * %.2f'%(
        my_inst_str,param_long['ndayslong'],param_long['ofact'],param_long['stthresh'],param_long['tfact'],
           param_long['athresh'],param_long['afact'])
    ax.set_ylabel("Uurbalans (GW of GWh)")
    ax.set_title(title)
#    figname = "../output/longst_io_"+my_inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
add_longst_io(landyrframe ,glb_inst_long,glb_inst_short,inst_str,axcmb[1,0],'green')    

cfigname = "../output/cmb_"+inst_str+'.png';
figcmb.savefig(cfigname,dpi=300) 
figcmb

sns.lineplot(data=landyrframe,x="validfrom",y="longtermst",ci=None) 

balansstats(landyrframe, "tolongterm",False,inst_str,axcmb[1,2])


#longststart
def add_longst_mem(df,my_inst_long,sizmul,my_inst_str,ax,color):
    param_long=param_longdf.to_dict('index')[my_inst_long]
    ststart = sizmul * param_long['ststart']
    df ["longtermsd" ] = memfunc(df ["tolongterm" ] . where(
          df ["tolongterm" ]<0,df ["tolongterm" ] *param_long['steff']),
          ststart,param_long['yrhalf']*365*24,0,4*ststart)  
    empty=df [df ["longtermsd" ] ==0 ].copy().reset_index()
    if empty.size !=0:
        print("WARNING: storage gets empty")
        print(empty["validfrom"])
    else:    
        print("OK: storage does not get empty")
    stoend= df.tail(1)["longtermsd"]
    if (stoend< ststart).any()  :
        print("WARNING: storage depleted over year %.0f < %.0f" % (stoend , ststart))
    else:        
        print("OK: storage surplus over year %.0f >%.0f" % (stoend , ststart))
    stomax=df["longtermsd"].max()
#    sns.lineplot(ax=ax,data=df,x="validfrom",y="longtermst",label="cum. balans",ci=None) 
    sns_plot=sns.lineplot(ax=ax,data=df,x="validfrom",y="longtermsd",label="storage",c=color,ci=None) 
    title= 'Long-time storage filling '+my_inst_str
    title = title +('\nstorage cycle eff %.0f %%, half-time %.1f yr\ninitial = %.0f max = %.0f end=%.0f'% (
        param_long['steff']*100,param_long['yrhalf'],ststart,stomax,stoend))
    ax.set_title(title)
    ax.set_ylabel("Opslag vulling (GWh)")
    ax.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.)
#    figname = "../output/longst_fill_"+my_inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
add_longst_mem(landyrframe ,glb_inst_long,1.0,inst_str,axcmb[1,1],'green')        


# +
### Korte termijn
# -

#dan moet de rest opgevangen door korte termijn
def add_shortst_io(df,my_inst_long,my_inst_short,my_inst_str,ax,color):
    param_short= param_shortdf.to_dict('index')[my_inst_short]
    print(param_short)

    shortsteff=param_short['steff']
    shortststart=param_short['ststart']
    df ["toshortterm" ] = df ["balans" ] -df ["tolongterm" ]
    df ["shorttermst" ] = df ["toshortterm" ] . where(
          df ["toshortterm" ]<0,df ["toshortterm" ] *shortsteff) .cumsum() +shortststart
    sns_plot=sns.lineplot(ax=ax,data=df,x="validfrom",y="toshortterm",color=color,ci=None) 
    #sns.lineplot(data=df,x="validfrom",y="shorttermst",ci=None) 
    param_long=param_longdf.to_dict('index')[my_inst_long]
    title= 'Short-time storage usage\nLong out smooth %.0f days, in if hour > %0d GW'%(
          param_long['ndayslong'],param_long['stthresh'])
    ax.set_title(title)
#    figname = "../output/shortst_io_"+my_inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
add_shortst_io(landyrframe ,glb_inst_long,glb_inst_short,inst_str,axcmb[2,0],'green')    

#opslag zonder halfwaardetijd en maxima
sns.lineplot(data=landyrframe,x="validfrom",y="shorttermst",ci=None) 

balansstats(landyrframe, "toshortterm",False,inst_str,axcmb[2,2])


# +
#nu opslag model toepassen
# -

def add_shortst_mem(df,my_inst_short,sizmul,my_inst_str,ax,color):
    param_short= param_shortdf.to_dict('index')[my_inst_short]
    shortsteff=param_short['steff']
    shortststart=sizmul*param_short['ststart']

    totverbruik = df['verbruik'].sum()
    #print('totverbruik %.1f'%(totverbruik))
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
    shortstend=  df.tail(1)["shorttermsd"] 
    if (shortstend< shortststart).any()  :
        print("WARNING: storage depleted over year %.0f < %.0f" % (shortstend , shortststart))        
#    plt.clf() 
    sns_plot=sns.lineplot(ax=ax,data=df,x="validfrom",y="shorttermsd",color=color,ci=None) 
    stomax=df["shorttermsd"].max()
    title= 'Short-time storage filling '+my_inst_str
    title = title +('\nstorage cycle eff %.0f %%, half-time %.0f days\ninitial = %.0f max = %.0f  end = %.0f'% (
        shortsteff*100,shorthalfw/24,shortststart,stomax,shortstend))
    ax.set_ylabel("Opslag vulling (GWh)")
    ax.set_title(title)
#    figname = "../output/shortst_fill_"+my_inst_str+'.png';
#    sns_plot.figure.savefig(figname,dpi=300) 
add_shortst_mem(landyrframe ,glb_inst_short,1.0,inst_str,axcmb[2,1],'green')        

figcmb.savefig(cfigname,dpi=300) 
figcmb

# +
#nieuwe manier van warmte apart
# -

figcmbw, axcmbw = plt.subplots(nrows=3, ncols=3,figsize=(10, 8))
figcmbw.tight_layout(pad=4)

landyrframe_e = landyrframe.copy()
landyrframe_e['balans'] = landyrframe_e['balans'] + 0.95*landyrframe_e['warmtevbr']
add_longst_io(landyrframe_e ,glb_inst_long,glb_inst_short,inst_str+'E',axcmbw[1,0],'blue')   
wlongtermfact=0.91
landyrframe_e ["tolongtermw" ] = ( wlongtermfact) *landyrframe_e ["tolongterm" ].where (
          landyrframe_e ["tolongterm" ] >0,0)
landyrframe_e ["tolongterm" ] = landyrframe_e ["tolongterm" ] - landyrframe_e ["tolongtermw" ]
#sns_plot=sns.scatterplot(data=landyrframe_e,x="validfrom",y="tolongterm") 
add_longst_mem(landyrframe_e ,glb_inst_long,0.25,inst_str+'E',axcmbw[1,1],'blue') 
landyrframe_e['balans'] = landyrframe_e['balans'] - landyrframe_e ["tolongtermw" ]

add_shortst_io(landyrframe_e ,glb_inst_long,glb_inst_short,inst_str+'E',axcmbw[2,0],'blue') 
add_shortst_mem(landyrframe_e ,glb_inst_short,1.0,inst_str+'E',axcmbw[2,1],'blue')   

landyrframe_w = landyrframe_e.copy()
landyrframe_w['balans'] = - landyrframe_w['warmtevbr']
#glb_inst_long_w='E'
wpmul=1
add_longst_io(landyrframe_w ,glb_inst_long,glb_inst_short,inst_str+'W',axcmbw[1,0],'red')    
landyrframe_w ["tolongterm" ] = landyrframe_w ["tolongterm" ] +wpmul*landyrframe_w ["tolongtermw" ] 
#sns_plot=sns.scatterplot(data=landyrframe_w,x="validfrom",y="tolongterm") 
add_longst_mem(landyrframe_w ,glb_inst_long,1.0,inst_str+'W',axcmbw[1,1],'red') 
landyrframe_w['balans'] = landyrframe_w['balans'] + wpmul*landyrframe_w ["tolongtermw" ]

add_shortst_io(landyrframe_w ,glb_inst_long,glb_inst_short,inst_str+'W',axcmbw[2,0],'red') 
add_shortst_mem(landyrframe_w ,glb_inst_short,0.25,inst_str+'W',axcmbw[2,1],'red')   

cfigwname = "../output/cmbw_"+inst_str+'.png';
figcmbw.savefig(cfigwname,dpi=300) 
figcmbw


# +
#oude manier van warmte

# +
### deel lange termijn opname voor warmte
def calclong_warmfrac(dfin,my_inst_long,my_inst_short,my_inst_str):
    df=dfin.copy()
    param_long=param_longdf.to_dict('index')[my_inst_long]
    print(param_long)
    nhrslong=24*param_long['ndayslong']
    balans1= df ["warmtevbr" ] 
    repyr=pd.concat ( [ balans1, balans1[0:nhrslong]  ] ).cumsum()
    bcalc= ( repyr.shift(-nhrslong)- repyr)/nhrslong
    df ["multdaywarmtevbr" ]= bcalc[0:len(df.index)]
    df ["multdaywarmtevbrsm" ]= np.convolve(df ["multdaywarmtevbr" ],
                                             np.ones(nhrslong)/nhrslong,mode='same' )
    df ["fromlongterm"] = - df ["tolongterm"].where(df ["tolongterm"]<0,0)
#    df ["longtermnaarwarmte"] = df ["multdaywarmtevbrsm" ]/ df ["fromlongterm"] 
    df ["longtermnaarwarmte"] = df ["fromlongterm"].where (df ["fromlongterm"]< df ["multdaywarmtevbrsm" ], 
                                                           df ["multdaywarmtevbrsm" ])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="validfrom",y= "fromlongterm",label="Totaal uit long",ax=ax1 )
    sns.scatterplot(data=df, x="validfrom",y= "longtermnaarwarmte" ,label="Als warmte uit long",ax=ax1)
    # Create and plotting the secondary y-axis data
    ax2 = ax1.twinx()
    df ["fromlongtermc"] =df ["fromlongterm"] .cumsum()
    df ["longtermnaarwarmtec"] =df ["longtermnaarwarmte"] .cumsum()
    sns.lineplot(data=df, x="validfrom",y= "fromlongtermc" ,label="Totaal uit long",ax=ax2,ci=None)
    sns.lineplot(data=df, x="validfrom",y= "longtermnaarwarmtec" ,label="Als warmte uit long",ax=ax2,ci=None)
    # Set labels with matching colors
    ax1.set_ylabel('Vermogen (GW/ GWh/hr)')
    ax2.set_ylabel('Cumulatieve energie (GWh)')
    ax1.legend( loc='upper left')
    ax2.legend( loc='upper center')
#    sns.scatterplot(data=df, x="validfrom",y= "longtermnaarwarmte" )
    
calclong_warmfrac(landyrframe ,glb_inst_long,glb_inst_short,inst_str)


# +
#nu run opnieuw met andere parameters
def run_again (cdf,my_inst_opw, my_inst_long,my_inst_short):
    my_inst_str=yrtomodel+my_inst_opw
    figcmbl, axcmbl = plt.subplots(nrows=3, ncols=3,figsize=(10, 8))
    figcmbl.tight_layout(pad=4)
    cdf= mkusopw(egasyr,yset7t0,yrtomodel,my_inst_opw)
    mkovplot(cdf,yrtomodel,my_inst_opw,my_inst_str,axcmbl[0,0]) 
#    plt.show()
    my_inst_str=yrtomodel+my_inst_opw+my_inst_long+my_inst_short
    cumbalplot(cdf,my_inst_str,axcmbl[0,1])    
    balansstats(cdf, "balans",True,my_inst_str,axcmbl[0,2])
    add_longst_io(cdf ,my_inst_long,my_inst_short,my_inst_str,axcmbl[1,0],'green'  )    
#    plt.clf()
    balansstats(cdf, "tolongterm",False,my_inst_str,axcmbl[1,2])
#    plt.clf()
    add_longst_mem(cdf ,my_inst_long,1.0,my_inst_str,axcmbl[1,1],'green' )   
#    plt.show()
#    plt.clf()
    add_shortst_io(cdf ,my_inst_long,my_inst_short,1.0,my_inst_str,axcmbl[2,0],'green')        
#    plt.clf()
    balansstats(cdf, "toshortterm",False,my_inst_str,axcmbl[2,2])
#    plt.clf()
    add_shortst_mem(cdf ,my_inst_short,my_inst_str,axcmbl[2,1],'green') 
    cfiglname = "../output/cmb_"+inst_str+'.png';
    figcmbl.savefig(cfiglname,dpi=300) 
    return figcmbl
    
run_again (landyrframe.copy(),glb_inst_opw,'D','B')  
# -

#regressietest op lang model A voor zo lang short model B zelfde parameters heeft als A
run_again (landyrframe.copy(),glb_inst_opw,'D','A')  

run_again (landyrframe.copy(),glb_inst_opw,'C','A')  


# +
def optlst(indf):
    lcol= indf.index.values+" - "+indf['omschr']
    return lcol.tolist()

def recalc(my2_inst_jaar,my2_inst_verbr,my2_inst_opw,my2_inst_long='B',my2_inst_short='A'):
    run_again (
       landyrframe.copy(),
       my2_inst_opw[0:1],
       my2_inst_long[0:1],
       my2_inst_short[0:1]  )
    
interact(
   recalc ,
       my2_inst_jaar=Dropdown(options=[yrtomodel], description='Data jaar:'),    
       my2_inst_verbr=Dropdown(options=optlst(param_verbr_df), description='Verbruik:'),    
       my2_inst_opw=Dropdown(options=optlst(param_opw_df), description='Opwek mix:'),
       my2_inst_long=Dropdown(options=optlst(param_longdf), description='Long-term:'),
       my2_inst_short=Dropdown(options=optlst(param_shortdf), description='Short-term:')                   
)
