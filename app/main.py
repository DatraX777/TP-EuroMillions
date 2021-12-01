from enum import Enum
from typing import Optional , List
from fastapi import FastAPI, Header
from pydantic import BaseModel


class Tirage(BaseModel):
    N1 : int
    N2 : int
    N3 : int
    N4 : int
    N5 : int
    E1 : int
    E2 : int

app = FastAPI()



@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model_name" : model_name, "message" : "deep Learning FTW!"}
    if model_name.value == "lenet":
        return {"model_name" : model_name , "message" : "LeCNN all the images"}
    return{"model_name" : model_name , "message" : "Have some residuals"}


class Item(BaseModel):
    name : str
    description: Optional[str] = None
    price : float
    tax : Optional[float] = None

@app.post("/api/predict/{tirage}")
async def est_gagnant(tirage: Tirage):
    _res : float

    return ("Proba gain : " + _res + "%, Proba perte : " + 1-_res)

@app.get("/api/predict/")
async def est_peut_etre_gagnant():
    tirage : Tirage
    return ("Ce tirage à de forte chance d'être gagnant :" + tirage)