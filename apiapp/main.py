from apiapp.request_class import ApiRequest
from fastapi import FastAPI

app = FastAPI()

from source.inference.agent import predict

@app.post("/request")
async def request_prediction(item: ApiRequest):
    spectra = item.spectra
    molform = item.molform
    print(spectra, molform)
    prediction = predict(molform, spectra)
    return {"prediction": prediction}