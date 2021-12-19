from enum import Enum
from typing import Optional , List
from typing_extensions import Required
from fastapi import FastAPI, Header
from pydantic import BaseModel
import random
from  api.mlModel import * 




class Tirage(BaseModel):
    date : str
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

@app.post("/api/predict/")
async def est_gagnant(tirage: Tirage) -> str:
    model = get_model()
    numbers : Tirage = tirage
    a,b = prediction(numbers,model)
    return ("Proba gain : " + b[1] + "%, Proba perte : " + b[0])

@app.get("/api/predict/")
async def est_peut_etre_gagnant() -> Tirage:
    model = get_model()
    tirage : Tirage = find_good_pick(model)
    return ("Ce tirage à de forte chance d'être gagnant : " + ''.join(str(e) + ' ' for e in tirage))

@app.get("api/model/")
async def model_spec() -> Model:
    return ("these are the model specs")


@app.put("/api/model/")
async def add_entry(item : Entry):
    if check_data_format(item):
        add_row_to_dataset(item)
        return("La données viens d'être ajouter au model")
    else:
        return("erreur, mauvais format de donnée")

@app.post("/api/model/retrain/")
async def retrain_model() -> dict:
    model = train_random_forest()
    save_model(model)
    a = loads_model_metrics()
    return (a)