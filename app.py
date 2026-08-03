from flask import Flask ,request ,jsonify ,render_template 
from flask_cors import CORS 
from predictor import SentimentEmotionPredictor 


app =Flask (__name__ )
CORS (app )

print ("Initializing AI Engine...")

engine =SentimentEmotionPredictor ()

@app .route ("/")
def home ():

    return "Urdu Sentiment & Emotion AI Backend is Running!"

@app .route ("/predict",methods =["POST"])
def predict ():
    try :

        data =request .get_json ()

        if not data or "text"not in data :
            return jsonify ({"error":"No text provided. Please send JSON with a 'text' field."}),400 

        text =data ["text"]


        result =engine .predict (text )

        return jsonify (result ),200 

    except Exception as e :
        return jsonify ({"error":str (e )}),500 

if __name__ =="__main__":
    print ("Starting Flask Web Server...")

    app .run (host ="0.0.0.0",port =5000 ,debug =True ,use_reloader =False )
