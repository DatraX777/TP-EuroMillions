import numpy as np
from numpy.core.defchararray import split
from numpy.core.fromnumeric import reshape, shape
from numpy.core.numeric import tensordot
import pandas as pd
from pandas.core.algorithms import mode
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import date , timedelta, datetime
import random
from joblib import dump, load

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
    return None

def train_random_forest(path):
    """entraine le modèle random forest à partir d'un fichier csv
    input: path chemin vers le csv
    output: rfc le modèle"""
    data = pd.read_csv(path,delimiter=",")
    # convertit les dates en nombre de jour de différence avec la date actuelle
    datedata = data["Date"]
    today = date.today()
    datedata2 = []
    for i in range(len(datedata)):
        deltaday = today - date(*map(int, datedata[i].split('-')))
        datedata2.append(deltaday.days)
    # création et entrainement du modèle
    rfc = RandomForestClassifier()
    X = data[["N1","N2","N3","N4","N5","E1","E2"]]
    X.insert(0,"Date diff",datedata2)
    y = data["win"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return rfc

def save_model(model):
    """sauvegarde le modèle"""
    dump(model, "app/databases/model_saved.joblib")
    return None

def get_model():
    """charge le modèle"""
    return load("app/databases/model_saved.joblib")

def prediction(x,trf):
    """donne la classe prédite pour x celon le modèle donné
    input: x donnée à prédire, trf modèle de prédiction
    output: classe prédite, probabilités"""
    temporaire = date.today() - date(*map(int, x[0].split('-')))
    x[0]= temporaire.days
    return trf.predict([x]), trf.predict_proba([x])

def add_row_to_dataset(new_data):
    """ajoute une donnée au dataset"""
    new_data.to_csv("app/databases/dataset.csv",index=False, header = False, mode = 'a')
    return None

def check_data_format(x):
    """vérifie le format de donnée"""
    if 0<x[1]<51 and 0<x[2]<51 and 0<x[3]<51 and 0<x[4]<51 and 0<x[5]<51 and 0<x[6]<13 and 0<x[7]<13:
        return True
    else :
        return False

def generate_random_data():
    """génère une donnée aléatoire à la date d'aujourd'hui"""
    random_data = []
    random_data.append(datetime.today().strftime('%Y-%m-%d'))
    random_data.extend(np.random.choice(range(1,50), 5, replace=False))
    random_data.extend(np.random.choice(range(1,12), 2, replace=False))
    return random_data

def find_good_pick(model, n = 1000):
    """trouve un tirage avec une forte probabilité de gain
    input: model modèle utilisé, n nombre d'itération max
    output: x au format ["Date","N1","N2","N3","N4","N5","E1","E2"]
    nota bene: la sortie "Date" indique le nombre de jour d'écart avec la date actuelle, soit 0 car tout les nouveau tirages sont prit pour aujourd'hui"""
    best_x = []
    best_b = [[0,0]]
    for i in range(n):
        x = generate_random_data()
        a, b = prediction(x, model)
        if b[0][1]>=0.7:
            return x, b
        else:
            if b[0][1]>best_b[0][1]:
                best_x = x
                best_b = b
    return best_x, best_b

path = 'app/databases/EuroMillions_numbers.csv'
data = import_data(path)

print(data)

data_size = data.shape[0]
print(data_size)

loosers = create_loosing_data(data_size,4)
print(loosers)

writing_dataset(data,loosers)

trained_random_forest = train_random_forest("app/databases/dataset.csv")
x = [[26,12,13,14,15,16,2,3]]
a, b = prediction(x,trained_random_forest)
print(a)
print(b)
# save_model(trained_random_forest)

model_saved = get_model()
a1,b1 = find_good_pick(model_saved)
print(a1)
print(b1)