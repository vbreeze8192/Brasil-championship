#Data analysis
import os
import io
import pandas as pd
import pickle 
import numpy as np
from datetime import datetime, date,timedelta
import streamlit as st

#custom functions
from BeRichFunctions import bestclassifier,train,save_pickle,team_metrics,champions_metrics,d_in_future, confMatrix

#
def download_excel(dftoexc,name_exc='Download_Excel'):
    # buffer to use for excel writer
    buffer = io.BytesIO()
    st.write(dftoexc)
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        dftoexc.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()
    st.download_button(
    label="Download data as Excel",
    data=buffer,
    file_name='{}.xlsx'.format(name_exc),
    mime='application/vnd.ms-excel')
        
    
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)



def doyourstupidthings(name,year_col,col_day,anni,anno_val,output_choice,day='NA'):
    input=['AVG_D_3Y_CH',\
            'AVG_D_N_CH',\
            'AVG_ND_3Y_S',\
            'AVG_ND_N_S',\
            'AVG_Dxd_3Y_CH',\
            'AVG_Dxd_N_CH',\
            'QTY_ND_3Y_S',\
            'QTY_ND_N_S',\
            'HOUR',\
            'HoA']

    original=pd.DataFrame()
    for anno in anni+[anno_val]:
        anno=str(anno)
        temp=pd.read_excel(name,sheet_name=anno)
        original=pd.concat([original,temp])


    st.write(':crown: Dati importati, bitches! :crown:')
    #crea colonna D con i pareggi
    original['D']=original['Res'].copy()
    original['D'].iloc[original['Res']=='D']=1
    original['D'].iloc[original['Res']!='D']=0
    #print(original)

    #dividi le squadre: crea 2 dataset temporanei che contengono solo le singole squadre home e away e relativo risultato
    temp_a=original[[col_day,year_col,'Date','Time','Away','AG','D']]
    temp_h=original[[col_day,year_col,'Date','Time','Home','HG','D']]

    #cambia nome delle colonne 
    temp_a=temp_a.rename(columns={"Away": "SQUADRA", "AG": "N_GOAL"})
    temp_h=temp_h.rename(columns={"Home": "SQUADRA", "HG": "N_GOAL"})
    temp_a['HoA']='0'
    temp_h['HoA']='1'

    #UNISCI I DUE DF
    raw=pd.concat([temp_h, temp_a])
    st.write(':floppy_disk: Ho fatto il dataset completo. Ora estraggo i dati utili: squadre, pareggi, altre amenità.')
    #print(raw)
    #estrai lista delle squadre
    squadre=list(raw.groupby(['SQUADRA']).mean().index)
    
    #print(squadre)
    #dividi il df in 3 anni precedenti e anno presente. Capire se estrae numero o string

    #raw['Date']=pd.to_datetime(raw['Date'])
    #print(raw['Date'].iloc[0])
    #raw[year_col]=raw['Date'].dt.year

    raw['HOUR']=0
    for ii in raw.index:
        raw['HOUR'].iloc[ii]=int(raw['Time'].iloc[ii].hour)
    for col in ['HG','AG','Res']:
        raw[col]=0

    #st.write('Valori non validi: {}'.format(raw.isna().sum()))
    raw=raw.fillna(0)

    for col in [col_day, 'N_GOAL', 'D', 'HoA', year_col]:
        raw[col]=raw[col].astype(int)

    st.write(':calendar: Divido il dataset completo in anni prima e anno corrente.')

    df3yr=pd.DataFrame()
    for anno in anni:
        df3yr=pd.concat([df3yr,raw[raw[year_col]==anno]])
    st.write(':white_check_mark: Nel passato ci sono queste squadre:')
    st.write(list(df3yr.groupby(['SQUADRA']).mean().index))

    st.write('Righe degli anni prima: {}'.format(df3yr.shape[0]))

    #Metriche di campionato
    st.write(':trophy: Calcolo le metriche per la championship e per singola squadra.')
    [avg3yrch,avgdxd3yrch]=champions_metrics(df3yr,col_day=col_day)
    #[nd3yrs,qtymax3yrs]=team_metrics(df3yr,squadre) 
    qty=pd.DataFrame(index=anni)
    ndr=pd.DataFrame(index=anni)
    for anno in anni:
        temp=df3yr[df3yr[year_col]==anno]
        [ndt,qtyt]=team_metrics(temp,squadre)

        ndt=ndt.reset_index()
        qtyt=qtyt.reset_index()
        qtyt=qtyt.rename(columns={"index": "SQUADRA"})
        qty=pd.concat([qty,qtyt]) #concateno anno su anno
        ndr=pd.concat([ndr,ndt])
    nd3yrs=ndr.groupby('SQUADRA').mean() #questo dovrebbe fare la media per squadra
    qtymax3yrs=qty.groupby('SQUADRA').mean()
    #raw è la rappresentazione dell'anno in corso. 
    #sul singolo anno di validazione, valuta sia i gol nel futuro veri, sia la predizione fatta dal modello. 

    raw=raw[raw[year_col]==anno_val]
    st.write('Righe per questo anno: {}'.format(raw.shape[0]))
    raw=raw.sort_values(col_day)
    start_day=raw[col_day].iloc[0]
    giornate=raw.groupby(col_day).mean().index #tutte le giornate per fare predizione. 


    
    df=pd.DataFrame()
    st.write('	:hourglass_flowing_sand: Guardo i dati per giornata...')

    nn=1
        #in che giorno nella riga
    if day=='NA':
        day=raw[col_day].iloc[-1]
    
    #Per ora fa la valutazione solo sulla giornata che viene fornita. 
    for day_iter in range(day,day+nn):
        final_df=pd.DataFrame()
        int_df=pd.DataFrame()
        print('Valuto la giornata {}'.format(day_iter))
        df_period=raw[raw[col_day]<=day_iter] #il dataframe contiene il periodo da giornata 0 a adesso
        squadre_day=list(df_period.groupby(['SQUADRA']).mean().index)
        st.write(':white_check_mark: In questa giornata ci sono queste squadre:')
        st.write(squadre_day)
        [avgnowch,avgdxdnowch]=champions_metrics(df_period,col_day=col_day)
        [ndnows,qtymaxnows]=team_metrics(df_period,squadre)


        for squadra in squadre_day:

            line_team=raw[raw['SQUADRA']==squadra]
            
            #Media di pari negli ultimi 3 anni per campionato
            line_team[input[0]]=avg3yrch

            #Media di pari negli ultimi giorni da D=0 a D=now
            line_team[input[1]]=avgnowch
            try:
                #Media di non pari negli ultimi 3 anni per squadra
                line_team[input[2]]=nd3yrs.loc[squadra].values[0]
            except Exception as e:
                st.write(":red[Sto avendo problemi con la squadra {}. Forse non c'era nel training, oppure l'hai scritto male!]".format(squadra))

            #Media di non pari negli ultimi giorni da D=0 a D=now
            line_team[input[3]]=ndnows[squadra]

            #Media di pari per giornata negli ultimi 3 anni per campionato
            line_team[input[4]]=avgdxd3yrch

            #Media di pari per giornata negli ultimi giorni da D=0 a D=now
            line_team[input[5]]=avgdxdnowch

            #Quantity di non pari longer negli ultimi 3 anni
            line_team[input[6]]=qtymax3yrs.loc[squadra].values[0]

            #Quantity di non pari da D=0 a D=now
            line_team[input[7]]=qtymaxnows.loc[squadra].values[0]
            
            int_df=pd.concat([int_df,line_team])

        final_df=int_df[int_df[col_day]==day_iter] #final df contiene la sola riga del giorno x
        final_df=final_df.fillna(0)
        st.write('Ecco il dataset su cui faccio previsioni. Ho riempito i valori nulli con 0.')
        download_excel(final_df,'Pre-trained_dataset_Day{}'.format(day))

        dict_input={'AVG_D_3Y_CH':'Media di pareggi per championship, calcolata sulle stagioni precedenti',\
            'AVG_D_N_CH':'Media di pareggi per championship, calcolata sulla stagione attuale',\
            'AVG_ND_3Y_S':'Media di non-pareggi per squadra, calcolata sulle stagioni precedenti',\
            'AVG_ND_N_S':'Media di non-pareggi per squadra, calcolata sulla stagione attuale',\
            'AVG_Dxd_3Y_CH':'Media di pareggi per giornata per championship, calcolata sulle stagioni passate',\
            'AVG_Dxd_N_CH':'Media di pareggi per giornata per championship, calcolata sulla stagione presente',\
            'QTY_ND_3Y_S':'Media del periodo massimo per stagione di giornate consecutive senza pareggi per squadra, calcolata sulle stagioni precedenti',\
            'QTY_ND_N_S':'Quantità di giornate consecutive senza pareggi per squadra sulla stagione attuale',\
            'HOUR':'Ora della partita',\
            'HoA':'Indicazione su Home o Away (0: Away, 1: Home)'}

        st.write("\n:robot_face: E mo' predico. :robot_face:")
        st.write("""I modelli sono allenati in due versioni diverse sui dati delle squadre e della championship. Gli input utilizzati sono:""")
        st.write(dict_input)
        ##Modello: predizioni per output
        nome_modello= os.path.join(os.getcwd(), os.path.normpath('Modello_{}'.format(output_choice)))
        dict=pickle.load(open(nome_modello, 'rb'))
        alg=dict['Algorithm']
        final_df['{}_pred'.format(output_choice)]=alg.predict(final_df[input])
        final_df['{}_probA'.format(output_choice)]=alg.predict_proba(final_df[input])[:,0]
        final_df['{}_probB'.format(output_choice)]=alg.predict_proba(final_df[input])[:,1]
        final_df=final_df.dropna()

        st.title('Risultati per la giornata {}'.format(day_iter))
        st.subheader("""Prima versione""")
        st.write('Questi risultati sono ottenuti con modelli allenati su questi input:')
        st.write(input)
        final_df=final_df.sort_values('{}_probA'.format(output_choice))
        
        st.write('Valutando {}, nella giornata {} dovresti investire su: :moneybag:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[ii]
            pred=final_df['{}_pred'.format(output_choice)].iloc[ii]
            prob=final_df['{}_probB'.format(output_choice)].iloc[ii]
            oth=final_df['{}_probA'.format(output_choice)].iloc[ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))

        st.write("___________________________________________")
        st.write('Valutando {}, le squadre con meno probabilità di pareggiare nella giornata {} sono: :sloth:'.format(output_choice,day_iter))
        for ii in range(1,8):
            sq=final_df['SQUADRA'].iloc[-ii]
            pred=final_df['{}_pred'.format(output_choice)].iloc[-ii]
            prob=final_df['{}_probB'.format(output_choice)].iloc[-ii]
            oth=final_df['{}_probA'.format(output_choice)].iloc[-ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))
        st.write('___________________________________________')
        if final_df['{}_probB'.format(output_choice)].mean()==1:
            st.write('Mah, ste probabilità so tutte uguali a 1. Grazie al c:sparkles:...')



        
         ##Modello: predizioni per output, con meno input
        input_lower=['AVG_ND_3Y_S',\
        'AVG_ND_N_S',\
        'QTY_ND_3Y_S',\
        'QTY_ND_N_S',\
        'HOUR',\
        'HoA']
        
        nome_modello= os.path.join(os.getcwd(), os.path.normpath('Modello_{}_lower_input'.format(output_choice)))
        dict=pickle.load(open(nome_modello, 'rb'))
        alg=dict['Algorithm']
        final_df['{}_lp_pred'.format(output_choice)]=alg.predict(final_df[input_lower])
        final_df['{}_lp_probA'.format(output_choice)]=alg.predict_proba(final_df[input_lower])[:,0]
        final_df['{}_lp_probB'.format(output_choice)]=alg.predict_proba(final_df[input_lower])[:,1]
        final_df=final_df.dropna()

        st.subheader("""Seconda versione""")
        st.write('Questi risultati sono ottenuti con modelli allenati su questi input:')
        st.write(input_lower)
        final_df=final_df.sort_values('{}_lp_probA'.format(output_choice))
        
        st.write('Valutando {}, nella giornata {} dovresti investire su: :moneybag:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[ii]
            pred=final_df['{}_lp_pred'.format(output_choice)].iloc[ii]
            prob=final_df['{}_lp_probB'.format(output_choice)].iloc[ii]
            oth=final_df['{}_lp_probA'.format(output_choice)].iloc[ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))

        st.write("___________________________________________")
        st.write('Valutando {}, le squadre con meno probabilità di pareggiare nella giornata {} sono: :sloth:'.format(output_choice,day_iter))
        for ii in range(0,7):
            sq=final_df['SQUADRA'].iloc[-ii]
            pred=final_df['{}_lp_pred'.format(output_choice)].iloc[-ii]
            prob=final_df['{}_lp_probB'.format(output_choice)].iloc[-ii]
            oth=final_df['{}_lp_probA'.format(output_choice)].iloc[-ii]
            st.write('	:soccer: Squadra: **:blue[{}]**, probabilità di pareggio: {} %'.format(sq,np.floor(prob*100)))
        st.write('___________________________________________')
        if final_df['{}_lp_probB'.format(output_choice)].mean()==1:
            st.write('Mah, ste probabilità so di nuovo tutte uguali a 1. Grazie al c:sparkles:...')
        else:
            st.write('Qui almeno abbiamo probabilità diverse')

        
        df=pd.concat([df,final_df])
        
    return(df)



###MAIN###

# Contents of ~/my_app/pages/page_2.py

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

st.title('Brasile 2023')

st.subheader("""Cosa vorresti prevedere?""")
st.write('Il modello è allenato per il periodo 2012-2022 della championship del Brasile.')
#uploaded_file = st.file_uploader("Upload Excel to explore", type=".xlsx")
#path=os.getcwd()

#name = st.text_input("Nome e percorso del file", 'BRA_2012-2022hst_2023og_xVERO.xlsx')
year_col = st.text_input("Nome della colonna dell'anno", 'Season')
col_day = st.text_input("Nome della colonna della giornata", 'giornata')
start_year=st.text_input("Anno iniziale del dataset completo",2012)
start_year=int(start_year)
end_year=st.text_input("Anno finale del dataset completo",2022)
end_year=int(end_year)
anno_val=st.text_input("Anno in corso",2023)
anno_val=int(anno_val)
day=st.text_input('Giornata che si vuole predire',10)
day=int(day)
#Campionato brasile. Un anno per ogni foglio
#Alleno su tutti gli anni, passando tutti gli anni sia per la media che per la predizione. 

anni=[*range(start_year,end_year+1)] #esclude 'l'ultimo anno
#valido sul 2022 che non ha mai visto.


#col_raw=['Giornata','Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','FTR']
col_raw=['Country','League','Season','Date','Time','Home','Away','HG','AG','Res']

output_select = st.radio(
    "Su quanti giorni vuoi prevedere la cumulata dei pareggi?",
    ('1','3','4'))

output_choice = 'D_in_{}iter'.format(output_select)
#'D_in_1iter', 'D_in_2iter', 'D_in_3iter',
st.write('Il modello prevede la probabilità che una squadra faccia almeno un pareggio nelle prossime {} giornate.'.format(output_select))
outputs=['D_in_4iter','D_in_2iter','D_in_1iter']
uploaded_file = st.file_uploader("Carica excel", type=".xlsx")

if st.button('Prevedi for Braaasil',disabled=not uploaded_file, type='primary'):
    st.write(':leaves:')
    final_df=doyourstupidthings(uploaded_file,year_col,col_day,anni,anno_val,output_choice,day)
    
    st.write('Ecco i dati completi per la giornata {}.'.format(day))
    download_excel(final_df,name_exc='Prediction_Day{}'.format(day))
    st.balloons()

#######PROSSIMI SVILUPPI##########
## NEXT: PER I DATI CHE HO NEL PASSATO RENDERE DISPONIBILE VALIDAZIONE. XAI PER CAPIRE COME MAI. 
## SQUADRE CAMBIANO: 3 ALL'ANNO. 

##ANALISI
# QUALE DISTRIBUZIONE HA LA SINGOLA SQUADRA NEI PAREGGI QUEST'ANNO? QUANTO NEGLI ANNI PRECEDENTI?
# SOLO 3-4 ANNI NEL TRAINING
# PUNTEGGIO: DOVE SI TROVA LA SQUADRA RISPETTO ALLE ALTRE?
# PUNTEGGIO NEL PASSATO?
# QTY PAREGGI MASSIMI E MEDI!

####NUOVI INPUT
## INTEGRARE ANCHE I PUNTI DELLA SQUADRA GG PER GIORNO E LA MEDIA LA DEV STD E MAX MIN DI TUTTE LE ALTRE SQUADRE NELLA GIORNATA CORRENTE. 
## MEDIA E DEV STD DI PUNTI DELLA CHAMPIONSHIP E DI QUELLE PRECEDENTI
## E ANCHE MEDIA E DEV STD DEGLI ANNI PRECEDENTI DELLA STESSA SQUADRA.
## INFO SULLA DISTRIBUZIONE DEI PAREGGI NEL TEMPO. COME RAPPRESENTARLA? NON SOLO MAC DEI GIORNI MA MEDIA SU PARTITA E DEV STD SU PARTITA DI MAC NON PARI. 
## COPPIA DI SQUADRE: QUANDO PAREGGIANO INSIEME RENDILO. 1. INCROCIARE PROB PAREGGIO E MOSTRARE LE COPPIE PIù ALTE. 
## MODELLO CHE PREVEDE LA PROBABILITA' SHARP DI PAREGGIO AL GG 1, 2 3 O 4, NON LA SOMMA. 
## MODELLO PER COPPIE: CASA/FUORI 
## INTEGRARE MEDIA E DEV STD DELLA DISTRIBUZIONE DI GOAL NELLE ULTIME PARTITE PER SQUADRA PER COFRONTARE LE DISTRIBUZIONI 
## MODELLINO MULTICLASSE CHE PREVEDE TUTTI E QUATTRO. 