import numpy as np
from numpy.core.defchararray import split
from numpy.core.fromnumeric import reshape, shape
from numpy.core.numeric import tensordot
import pandas as pd
import sklearn as sk
from datetime import date , timedelta, datetime
import random

def import_data(path):
    """importe les données
    input: chemin vers les données
    output: les données"""
    data = pd.read_csv(path,delimiter=";")
    data2 = pd.DataFrame(data, columns=["Date","N1","N2","N3","N4","N5","E1","E2"])
    data2["win"]=[1]*data2.shape[0]
    return data2

def random_date_generator(enddate):
    """genere une date aleatoire comprise entre 01/01/2004 et enddate
    input: une date format str(yyyy-mm-dd)
    output: une date au format date(yyyy-mm-dd)"""
    start_date = date(2004, 1, 1)
    end_date = date(*map(int, enddate.split('-')))
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    return random_date

def create_loosing_data(size_of_data, loosing_rate, enddate = "2021-12-31"):
    """genere un set de donnees supposees perdantes
    input: size_of_data est la taille de données souhaitée, enddate est la date de fin pour la génération des nouveaux résultats
    output: loosing_data est un dataframe des données générées"""
    gen_date = []
    E=[]
    N=[]
    for i in range(size_of_data*loosing_rate):
        E.append(np.random.choice(range(1,12), 2, replace=False))
        N.append(np.random.choice(range(1,50), 5, replace=False))
        gen_date.append(random_date_generator(enddate))
    A = np.concatenate([N,E], axis=1)
    loosing_data = pd.DataFrame(A,columns=["N1","N2","N3","N4","N5","E1","E2"])
    loosing_data.insert(0,"Date",gen_date)
    loosing_data["win"]=[0]*size_of_data*loosing_rate
    return loosing_data

def writing_dataset(win_set, loose_set):
    """crée un fichier contenant le dataset composé des numéros gagnants et perdants
    input: win_set, loose_set -> dataset
    output: None"""
    frames = [win_set, loose_set]
    result = pd.concat(frames,ignore_index=True)
    result.to_csv("app/databases/dataset.csv",index=False)
    return 0 

path = 'app/databases/EuroMillions_numbers.csv'
data = import_data(path)

print(data)
data_size = data.shape[0]
print(data_size)
loosers = create_loosing_data(data_size,4)
print(loosers)
a = writing_dataset(data,loosers)