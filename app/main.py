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

class Model(BaseModel):
    Metrics: List[float]
    name : str
    trainMetrics : List[float]


app = FastAPI()

@app.post("/api/predict/")
async def est_gagnant(tirage: Tirage) -> str:
    _res : float
    numbers : Tirage = tirage
    return ("Proba gain : " + _res + "%, Proba perte : " + 1-_res)

@app.get("/api/predict/")
async def est_peut_etre_gagnant() -> Tirage:
    tirage : Tirage = [random.range(1,51),]
    return ("Ce tirage à de forte chance d'être gagnant :" + tirage)

@app.get("api/model/")
async def model_spec() -> Model:
    return ("these are the model specs")

