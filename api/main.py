from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL=tf.keras.models.load_model("D:\studymaterial\projects\Potato disease\saved_models\s")
CLASS_NAMES=["Early Blight","Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Alive"

def read_file_as_image(data)-> np.ndarray:
    #Image.open(BytesIO(data)) reads bytes of image as Pillow Image 
    #np.array(x) converts Pillow image 'x' to numpy array 
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile =File(...)):
    image = read_file_as_image(await file.read())
    
    #read_file_as_image function converts image to np array
    
    """np.expand_dims(image,0[axis =1 is column level])) makes it 2d since our predict function
    function in the notebook takes like [[255,255,3]]. dims converts [255,255,3]-> [[255,255,3]]"""
    img_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)

    index=np.argmax(predictions[0])
    predicted_class= CLASS_NAMES[index]

    confidence=np.max(predictions[0])

    print(predicted_class,confidence)

    return {'class':predicted_class,'confidence':float(confidence)}

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)