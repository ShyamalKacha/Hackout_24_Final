import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

savedModel=load_model('model1.h5')
df=pd.read_csv("label.csv")
lab={}
for i in zip(df["encoded"].tolist(),df["actual"].tolist()):
    lab[i[0]]=i[1]






def greet(nitrogen, Phosphorous, Potassium, temperature,humidity,ph,rainfall):
    x=np.array([nitrogen, Phosphorous, Potassium, temperature,humidity,ph,rainfall])
    x=x.reshape(-1,1)

    

    scaler = StandardScaler() 
    scaled_x = scaler.fit_transform(x) 
    # df2=pd.DataFrame(scaled_x)
    scaled_x=scaled_x.reshape(1,-1)
    scaled_x=scaled_x.tolist()

    temparr=np.array(scaled_x[0])
    temparr=temparr.reshape(1,-1)
    # temparr.tolist()

    pre=savedModel.predict(temparr)
    val=np.argmax(pre)
    return lab[val]
   
    

# demo = gr.Interface(
#     fn=greet,
#     inputs=["number", "number","number","number","number","number","number"],
    
#     outputs=["text"]

    
# )
# demo.launch()   

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            nitrogen = gr.Number(label="Nitrogen")
            Phosphorous = gr.Number(label="Phosphorous")
            Potassium = gr.Number(label="Potassium")
            temperature = gr.Number(label="temperature")
            humidity = gr.Number(label="humidity")
            ph = gr.Number(label="ph")
            rainfall = gr.Number(label="rainfall")
        with gr.Column(scale=1, min_width=100):
            output = gr.Textbox(label="Prediction")
            greet_btn = gr.Button("Submit")
            greet_btn.click(fn=greet, inputs=[nitrogen, Phosphorous, Potassium, temperature,humidity,ph,rainfall], outputs=output, api_name="greet")


demo.launch()