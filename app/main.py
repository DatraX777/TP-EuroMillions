from enum import Enum
from typing import Optional , List
# from typing_extensions import Required
from fastapi import FastAPI, Header
from pydantic import BaseModel
import random
from  api.mlModel import * 
import pickle
import datetime




class Tirage(BaseModel):
    N1 : int
    N2 : int
    N3 : int
    N4 : int
    N5 : int
    E1 : int
    E2 : int

class Model(BaseModel):
    Metrics: List[float]
    name : str
    trainMetrics : List[float]

class Entry(BaseModel):
    date: str
    tirage : Tirage
    win : int 
    gain : int


app = FastAPI()

@app.get("/api/model")
async def initialiser():
    init()
    return ("Le modèles viens d'être générer")

@app.post("/api/predict/")
async def est_gagnant(tirage: Tirage):
    model = get_model()
    numbers = Tirage_to_list(tirage)
    a,b = prediction(numbers,model)
    return ("Proba gain : " + str(b[0][1]) + "%, Proba perte : " + str(b[0][0]))

@app.get("/api/predict/")
async def est_peut_etre_gagnant():
    model = get_model()
    tirage, proba_win = find_good_pick(model)
    #print(tirage)
    return {"Ce tirage à de forte chance d'être gagnant ": str(tirage), "proba win": proba_win[0][1]}

@app.get("/api/model/")
async def model_spec():
    print("tata")
    specs :dict = loads_model_metrics()
    print("toto")
    return (specs)


@app.put("/api/model/")
async def add_entry(item : Entry):
    if check_data_format(item):
        add_row_to_dataset(item)
        return("La données viens d'être ajouter au model")
    else:
        return("erreur, mauvais format de donnée")

@app.post("/api/model/retrain/")
async def retrain_model():
    model = train_random_forest()
    save_model(model)
    return ("model successfully retrained")

def Tirage_to_list(tirage):
    listTirage = [tirage.date, tirage.N1, tirage.N2, tirage.N3, tirage.N4, tirage.N5, tirage.E1, tirage.E2]
    return(listTirage)