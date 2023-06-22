#Data analysis
import pandas as pd
import os
import pickle 
import numpy as np

#Modeling
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import r2_score



#Funzioni per allenare il modello 
def rfc(X_train, y_train,X_test,y_test):
    param_dist = {'n_estimators': np.random.randint(50,high=500),
            'max_depth': np.random.randint(1,high=20)}

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, 
                                param_distributions = param_dist, 
                                n_iter=5, 
                                cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)

    
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    # Generate predictions with the best model
    y_pred = best_rf.predict(X_test)
    confMatrix(y_test, y_pred)


    return(best_rf,rand_search.score(X_test, y_test))

def confMatrix(y_test,y_pred,title='Confusion Matrix'):
          # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp =ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    disp.ax_.set_title(title)

def bestclassifier(X_train, y_train,X_test,y_test,gridsearch=True):

    """Finds the best pipeline for the data provided within classifier, using gridsearch or randomizedsearch""" 
    # Initialize the estimators
    clf1=ExtraTreesClassifier(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    #clf3=XGBClassifier(random_state=42,gamma=0,scale_pos_weight=1,validation_fraction=0.1)
    
    #Algorithm settings
    settings1={'classifier__n_estimators':[50,100,150,200],'classifier__max_depth':[3,4], 'classifier__min_samples_split':[3, 5, 7],'classifier':[clf1]}
    settings2 = {'classifier__n_estimators':[40,50,70,80,100], 'classifier__max_depth':[3,4], 'classifier__min_samples_split':[3, 5, 7],'classifier':[clf2]}
    #settings3 = {'classifier__max_depth':[2,3,5], 'classifier__min_child_weight':[2,3,5],'classifier__n_estimators':[50,100,150],\
     #   'classifier__learning_rate':[0.5,0.2,0.1,0.05],'classifier':[clf3]}

    #Final pipeline
    params = [settings1, settings2]#, settings3]
    pipe = Pipeline([\
        ('scl', StandardScaler()),\
        ('classifier', clf1)])

    #Model search
    if gridsearch==True:
        #With gridsearch:
        gs=GridSearchCV(pipe, params, cv=3, n_jobs=-1,verbose=-1).fit(X_train, y_train) 
    else:
        #With Random search:
        gs=RandomizedSearchCV(pipe, params, cv=3, n_jobs=-1, verbose=-1).fit(X_train, y_train)
    
    print('Found best estimator')
    my_params = gs.best_params_
    print('Best algorithm:\n', gs.best_params_['classifier'])
    print('Best params:\n', gs.best_params_)
    alg=gs.best_estimator_
    prediction = gs.predict(X_test)
    for ii in range(0,10):
        print('P:{}, R:{}'.format(prediction[ii],y_test[ii]))
    
    # Create confusion matrix
    print('Confusion matrix: {}'.format(confusion_matrix(y_test, prediction)))
    # Display accuracy score
    print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
    # Display F1 score
    #print('F1 Score: {}'.format(f1_score(y_test,prediction)))
    

    print('GridSearchCV accuracy:\n', gs.score(X_test, y_test))
    return(alg,gs.score(X_test, y_test))



    


def train(df,inputs, OUTPUT,task='reg',testsize=0.25,path=os.getcwd(),nome_modello='NA'):
    """Uses best model to train a model"""
    #TODO: define what's test, validation, train
    ordir=os.getcwd()
    df_train,df_test = train_test_split(df,test_size=testsize)

    X_train = df_train[inputs]
    y_train = df_train[OUTPUT]
    X_test = df_test[inputs]
    y_test = df_test[OUTPUT]
    output_mean = y_test.mean()

    if task=='reg':
        [alg,score_alg]=bestregressor(X_train, y_train,X_test, y_test)
    elif task=='clf':
        [alg,score_alg]=bestclassifier(X_train, y_train,X_test, y_test)
    elif task=='rfc':
        [alg,score_alg]=rfc(X_train, y_train,X_test, y_test)
    else:
        raise Exception('Task must be reg for regression and clf for classification. rfc for random foresst classifier.')
    os.chdir(path)
    (dicts,nome_modello)=save_pickle(alg,inputs,OUTPUT,\
    score_alg)
    os.chdir(ordir)
    return(alg,dicts,nome_modello)

def save_pickle(alg,inlist,outlist,score,nome_modello='NA'):
    """Saves pickle in quasi-standard structure for Rebecca"""
    if nome_modello=='NA':
        nome_modello='Modello_{}'.format(outlist)
    save={}
    save['Algorithm']=alg
    save['Input']=inlist
    save['Output']=outlist
    save['Score_test']=score
    pickle.dump(save, open(nome_modello, 'wb'))
    pickle.dump(save['Algorithm'], open('{}_concrete_algorithm'.format(nome_modello), 'wb'))
    print('Saved')
    return(save,nome_modello)


#funzioni per estrarre le info da utilizzare in train e test?
#indicatori da champion 3 yr prec: da calcolare sul raw o sul primo livello di tabella? Rischi oche si raddoppi? non dovrebbe sulle medie
def team_metrics(df,squadre,col_squadre='SQUADRA',col_d='D'):
        '''Calcola le metriche per campionati precedenti'''

        #Media di non pari nel periodo considerato per squadra: df con index=squadra, value=media
        nd_s=df.groupby([col_squadre]).mean()[col_d] 

        #Quantity di non pari longer nel periodo considerato: df con index=squadra, value=media
        qtymax_s=pd.DataFrame(index=squadre,columns=['qtymax_s'])

        ii=0
        
        for squadra in squadre:
                #Itero su tutte le squadre per creare il df
                maxx=0
                temp=df[df[col_squadre]==squadra]
                #itero su tutte le osservazioni per trovare il massimo di d consecutivi
                for item in range(0,len(temp)):
                        if temp[col_d].iloc[item]==0:
                                ii=ii+1
                                if ii>maxx:
                                        maxx=ii
                                        ii=0
                        
                qtymax_s.loc[squadra]=maxx
        
        return(nd_s,qtymax_s)

def champions_metrics(df,col_d='D',col_day='Giornata'):
        '''Calcola le metriche per campionati precedenti'''
        #Media di pari nel periodo considerato per campionato: valore singolo
        avg_ch=df[col_d].mean() #da capire se ok

        #Media di pari per giornata nel periodo considerato per campionato: valore singolo
        avgdxd_ch=df.groupby([col_day]).mean().mean()[col_d] #primo mean df con medie per ogni giorno, secondo mean media su tutti i giorni

        
        return(avg_ch,avgdxd_ch)


def d_in_future(df,Nmax,col_d='D'):
        out_list=[]
        for N in range(1,Nmax+1):
                out_list=out_list+['{}_in_{}iter'.format(col_d,N)]
                df['{}_in_{}iter'.format(col_d,N)]=0
                for ii in range(0,len(df.index)-N):
                        start=df.index[ii]
                        stop=df.index[ii+N] #da capire se viene nelle successive 3 o 4
                        if df[col_d].iloc[start:stop].sum()>0:
                                df['{}_in_{}iter'.format(col_d,N)].loc[start]=1
                df['{}_in_{}iter'.format(col_d,N)]=df['{}_in_{}iter'.format(col_d,N)].astype(int)	
        #eventualmente aggiungere colonna aggiuntiva per numero di partita di x
        return(df,out_list)

