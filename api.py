# Import des librairies
from fastapi import FastAPI
from fastapi import File, UploadFile
from pydantic import BaseModel
import uvicorn, pickle

tags = [
    {
        "name": "hello",
        "description": "Operations related to hello",
    },
    {
        "name": "predict",
        "description": "Operations related to predict",
    },
]


# Création de l'application
app = FastAPI(
    title="API de prédiction de défaut de paiement",
    description="API permettant de prédire si un client va faire défaut de paiement",
    version="1.0.0",
    openapi_tags=tags
)

# Point de terminaison standard
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Point de terminaison avec paramètre
@app.get("/hello_you")
async def hello_you(name: str):
    return {"message": f"Hello {name}"}

# Point de terminaison avec paramètre optionnel
@app.get("/hello_you/{name}")
async def hello_you(name: str):
    return {"message": f"Hello {name}"}

class Credit(BaseModel):
    age: int
    job: int
    marital: int
    education: int
    default: int
    balance: int
    housing: int
    loan: int
    campaign: int
    pdays: int
    previous: int
    poutcome: int


# Point de terminaison : Prédiction
@app.post("/predict")
def predict(data: Credit):
    # Chargement du modèle
    model = pickle.load(open("model.pkl", "rb"))
    print(True)
    # Prédiction
    prediction = model.predict([list(data.dict().values())])
    # Renvoi de la prédiction
    return {"prediction": int(prediction[0])}


# Point de terminaison qui permet de verser un fichier
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)