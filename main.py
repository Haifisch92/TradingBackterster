logo = """
    ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗     ██████╗  █████╗  ██████╗██╗  ██╗████████╗███████╗███████╗████████╗███████╗██████╗ 
    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝     ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗
       ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗    ██████╔╝███████║██║     █████╔╝    ██║   █████╗  ███████╗   ██║   █████╗  ██████╔╝
       ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║    ██╔══██╗██╔══██║██║     ██╔═██╗    ██║   ██╔══╝  ╚════██║   ██║   ██╔══╝  ██╔══██╗
       ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝    ██████╔╝██║  ██║╚██████╗██║  ██╗   ██║   ███████╗███████║   ██║   ███████╗██║  ██║
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝"""
                                                                                                                                             


import pandas
import time
import re
import numpy
import os
import argparse
from itertools import groupby
import math
from fuzzywuzzy import process


def _split_list(l):

    l1 = l[:math.ceil(len(l)/2)]
    l2 = l[math.ceil(len(l)/2):]

    return (l1,l2)

def _fill_list(l1,l2,val=None):

    lmin =  l1 if len(l1) < len(l2) else l2
    lmax =  l1 if l2 is lmin else l2

    for x in range(0,abs(len(l1)-len(l2))):

            lmin.append(val)

    return (lmax,lmin)

def _round(fl):
    frac, whole = math.modf(fl)
    if frac > 0.5:
        return math.ceil(fl)
    else:
        return math.floor(fl)


def stoch(df):
    k_period = 14
    d_period = 3

    n_height = df['High'].rolling(k_period).max()
    n_low = df['Low'].rolling(k_period).min()
    K = (df['Close'] - n_low) * 100 / (n_height - n_low)
    df['K'] = K.rolling(3).mean()
    df['D'] = df['K'].rolling(d_period).mean()
    return df


def ma(df,periodo):
    df[f'ma{periodo}'] = df["Close"].rolling(periodo).mean()
    return df

def time_group(df,delta):
    """
    Raggruppa gli elementi del Dataframe per date consecutive controllandole
    con un time delta in stringa es: 1m , 1h ...
    """
    dt = df['Timestamp']
    day = pandas.Timedelta(delta)

    in_block = (dt.diff(-1) == -day) | (dt.diff() == day)
    filt = df.loc[in_block]
    breaks = filt['Timestamp'].diff() != day
    groups = breaks.cumsum()
    section = None
    for _, frame in filt.groupby(groups):
        section = frame
    return section


def query_df(df,query=None):
    """Lavora sul Dataframe , carica gli indicatori e
       controlla che rispettino le query, controlla se il gruppo che
       rispetta la query faccia riferimento all ultima raw del DF principale"""

    # Carico le medie mbili , lo stocastico e il numero di candele consegutive

    ultima_candela = df['Timestamp'].iloc[-1]
    df = ma(df,5)
    df = ma(df,10)
    df = ma(df,20)
    df = ma(df,60)
    df = ma(df,123)
    df = ma(df,233)
    df = ma(df,300)
    df = stoch(df)
    df = count_candelstick(df)
    df = count_iperstoch(df)

    # Controlla la query e divide in gruppi dove vengono rispettati i valori
    if query != None:
        df = df.query(query)
        df = df.reset_index(drop=True)
        return df
        df = time_group(df,registro[tf])

        #Se trova corrispondenza con l'ultima manda il segnale
        if ultima_candela == df['Timestamp'].iloc[-1]:
            return "Trovata corrispondenza"

    return df


def count_candelstick(df):
    """Assegna alla colonna candela il valore rosso/verde e
       conta quante sono consecutive"""

    df['Candela'] = numpy.where((df['Close'] > df['Open']) , 'Verde', 'Rossa')
    df["Num_candele"] =  (df['Candela'].groupby((df['Candela'] != df['Candela'].shift()).cumsum()).cumcount() + 1)
    return df

def count_iperstoch(df):

    conditions  = [ (df['K'] >= 80) & (df['D'] >= 80), (df['K'] <= 20) & (df['D'] <= 20) ]
    choices     = [ "Ipercomprato", 'Ipervenduto' ]

    df["Stoch"] = numpy.select(conditions, choices, default='nel_canale')
    df['Num_iper'] = (df['Stoch'].groupby((df['Stoch'] != df['Stoch'].shift()).cumsum()).cumcount() + 1)
    return df


def standard_csv(df):
    columns = list(df.columns)
    default_values = ['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    new_col = [process.extractOne(val, default_values)[0] if process.extractOne(val, default_values)[1] > 60 else val for val in columns ]
    df.columns = new_col

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--Output",help="output file")
    parser.add_argument("-f","--Input",help="input file")
    parser.add_argument("-g","--Grafico",help="modalità grafica",action='store_true')
    parser.add_argument("-m","--tabella_movimenti",help="tabella movimenti")
    args = parser.parse_args()

    if args.Input == None:
        raise ValueError("Devi specificare un file di input")

    df = standard_csv(pandas.read_csv(args.Input))
    df = query_df(df)
    mov = args.tabella_movimenti


    if mov != None:
        df_mov = query_df(df,query=f"Close - Open >= {mov} or Open - Close >= {mov}")
        trace = {}
        for inx,x in df_mov.iterrows():
            if x['Timestamp'] in trace:
                trace[x['Timestamp']]['count'] += 1
            else:
                trace[x['Timestamp']] = {'count':1}



    df_verdi = query_df(df,query="Candela == 'Verde'")
    candeleV = df_verdi.iloc[df_verdi['Num_candele'].idxmax()]
    val_candeleV = str(candeleV).split()[:-4]
    candeleV_pulito = [ df_verdi['Num_candele'].iloc[idx-1] for idx,x in df_verdi['Num_candele'].items() if df_verdi['Num_candele'].iloc[idx-1] > x ]
    group_candeleV = [(el, len(list(group))) for el, group in groupby(sorted(candeleV_pulito))]
    tot = sum([frequenza for valore,frequenza in group_candeleV])
    m = [(val,round(((freq / tot) * 100),2))for val,freq in group_candeleV]
    mp_candeleV = _round(sum([valore*frequenza for valore,frequenza in m])/100)
    val_candeleV = val_candeleV + ['Mp Num_candele',mp_candeleV]

    df_rosse = query_df(df,query="Candela == 'Rossa'")
    candeleR = df_rosse.iloc[df_rosse['Num_candele'].idxmax()]
    val_candeleR = str(candeleR).split()[:-4]
    candeleR_pulito = [ df_rosse['Num_candele'].iloc[idx-1] for idx,x in df_rosse['Num_candele'].items() if df_rosse['Num_candele'].iloc[idx-1] > x ]
    group_candeleR = [(el, len(list(group))) for el, group in groupby(sorted(candeleR_pulito))]
    tot = sum([frequenza for valore,frequenza in group_candeleR])
    m = [(val,round(((freq / tot) * 100),2))for val,freq in group_candeleR]
    mp_candeleR = _round(sum([valore*frequenza for valore,frequenza in m])/100)
    val_candeleR = val_candeleR + ['Mp Num_candele',mp_candeleR]

    df_ipercomprato = query_df(df,query="Stoch == 'Ipercomprato'")
    ipercomprato = df_ipercomprato.iloc[df_ipercomprato['Num_iper'].idxmax()]
    val_ipercomprato = str(ipercomprato).split()[:-4]
    ipercomprato_pulito = [ df_ipercomprato['Num_iper'].iloc[idx-1] for idx,x in df_ipercomprato['Num_iper'].items() if df_ipercomprato['Num_iper'].iloc[idx-1] > x ]
    group_ipercomprato = [(el, len(list(group))) for el, group in groupby(sorted(ipercomprato_pulito))]
    tot = sum([frequenza for valore,frequenza in group_ipercomprato])
    m = [(val,round(((freq / tot) * 100),2))for val,freq in group_ipercomprato]
    mp_ipercomrato = _round(sum([valore*frequenza for valore,frequenza in m])/100)
    val_ipercomprato = val_ipercomprato + ['Mp Num_iper',mp_ipercomrato]

    df_ipervenduto = query_df(df,query="Stoch == 'Ipervenduto'")
    ipervenduto = df_ipervenduto.iloc[df_ipervenduto['Num_iper'].idxmax()]
    val_ipervenduto = str(ipervenduto).split()[:-4]
    ipervenduto_pulito = [ df_ipervenduto['Num_iper'].iloc[idx-1] for idx,x in df_ipervenduto['Num_iper'].items() if df_ipervenduto['Num_iper'].iloc[idx-1] > x ]
    group_ipervenduto = [(el, len(list(group))) for el, group in groupby(sorted(ipervenduto_pulito))]
    tot = sum([frequenza for valore,frequenza in group_ipervenduto])
    m = [(val,round(((freq / tot) * 100),2))for val,freq in group_ipervenduto]
    mp_ipervenduto = _round(sum([valore*frequenza for valore,frequenza in m])/100)
    val_ipervenduto = val_ipervenduto + ['Mp Num_iper',mp_ipervenduto]

    if args.Grafico:
        print(logo)
        print("+","-"*149,"+")

        print("|"," "*5,"Backtest Candele Verdi"," "*15,"Backtest Candele Rosse"," "*15,"Backtest Ipercomprato"," "*15,"Backtest Ipervenduto"," "*6,"|")
        for x in range(0,len(val_candeleV),2):

            stringhe = ["|",val_candeleV[x],val_candeleV[x+1],"|",
                        "|",val_candeleR[x],val_candeleR[x+1],"|",
                        "|",val_ipercomprato[x],val_ipercomprato[x+1],"|",
                        "|",val_ipervenduto[x],val_ipervenduto[x+1],"|"]

            print("{:<2} {:<15}  {:>15} {:>2} {:<2} {:<15} {:>15} {:>2} {:<2} {:<15}  {:>15} {:>2} {:<2} {:<15} {:>15} {:>2}".format(*stringhe))

        print("+","-"*149,"+")
        if mov != None:
            print("|"," "*53,f"Conteggio orario con movimento candele: {mov}$"," "*52,"|")
            t1,t2 = _split_list(sorted(trace))
            t1,t2 = _fill_list(t1,t2,val="")

            for v,x in zip(t1,t2):
                if x != "":
                    print("{:<58} {:<5} {:>5}     {:<5} {:>5} {:>60} ".format("|",v,trace[v]['count'],x,trace[x]['count'],"|"))
                else:
                    conteggio2 = ""
                    print("{:<58} {:<5} {:>5} {:>79} ".format("|",v,trace[v]['count'],"|"))
            print("+","-"*149,"+")
    else:
        if mov != None:
            t1,t2 = _split_list(sorted(trace))
            t1,t2 = _fill_list(t1,t2,val="")

            for v,x in zip(t1,t2):
                if x != "":
                    print("{:<2} {:<5}  {:>5} {:>2}".format(v,trace[v]['count'],x,trace[x]['count']))
                else:
                    print("{:<2} {:<5}".format(v,trace[v]['count']))
            
        print("Numero massimo candele verdi consecutive:",candeleV["Num_candele"])
        print("Media ponderata sulle candele verdi consecutive:",mp_candeleV)

        print("Numero massimo candele rosse consecutive:",candeleR["Num_candele"])
        print("Media ponderata sulle candele rosse consecutive:",mp_candeleR)

        print("Numero massimo candele in ipercomprato:",ipercomprato['Num_iper'])
        print("Media ponderata sulle candele in ipercomprato consecutive:",mp_ipercomrato)

        print("Numero massimo candele in ipervenduto:",ipervenduto['Num_iper'])
        print("Media ponderata sulle candele in Ipervenduto consecutive:",mp_ipervenduto)

    
    

    if args.Output != None:
        df.to_csv(args.Output,index=False)




