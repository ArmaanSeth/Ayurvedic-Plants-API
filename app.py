import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

model = tf.keras.models.load_model("model1 .h5")


def predict(image: Image.Image):
    image = image.resize((256, 256))
    image = np.asfarray(image) / 255
    image = np.expand_dims(image, 0)
    result = np.argmax(model.predict(image), axis=1)
    return result


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

app = FastAPI()

@app.get('/')
async def hello_world():
    return "hello world"
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    print("t")
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)[0]
    print(type(prediction))
    return {"id":int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=8000,reload=True)
