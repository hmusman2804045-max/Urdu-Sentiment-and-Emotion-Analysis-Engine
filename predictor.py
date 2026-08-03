import os 
import torch 
import numpy as np 
from transformers import AutoTokenizer ,AutoModelForSequenceClassification 

class SentimentEmotionPredictor :
    def __init__ (self ):

        base_dir =os .path .dirname (os .path .abspath (__file__ ))
        self .sentiment_dir =os .path .join (base_dir ,"models","sentiment_model")
        self .emotion_dir =os .path .join (base_dir ,"models","emotion_model")


        self .sentiment_map ={0 :"Negative",1 :"Neutral",2 :"Positive"}
        self .emotion_map ={0 :"Joy",1 :"Anger",2 :"Fear",3 :"Sadness"}

        self .tokenizer =None 
        self .sentiment_model =None 
        self .emotion_model =None 

        self .load_models ()

    def load_models (self ):
        print ("Loading Tokenizer...")
        self .tokenizer =AutoTokenizer .from_pretrained (self .sentiment_dir )

        print ("Loading Sentiment Model...")
        self .sentiment_model =AutoModelForSequenceClassification .from_pretrained (self .sentiment_dir )
        self .sentiment_model .eval ()

        print ("Loading Emotion Model...")
        self .emotion_model =AutoModelForSequenceClassification .from_pretrained (self .emotion_dir )
        self .emotion_model .eval ()

        print ("Models loaded into memory successfully!")

    def predict (self ,text ):
        if not text or not str (text ).strip ():
            return {"error":"Empty text provided."}


        inputs =self .tokenizer (text ,return_tensors ="pt",truncation =True ,max_length =128 )


        with torch .no_grad ():
            sentiment_out =self .sentiment_model (**inputs ).logits 
            emotion_out =self .emotion_model (**inputs ).logits 


        sentiment_idx =np .argmax (sentiment_out .numpy (),axis =-1 )[0 ]
        emotion_idx =np .argmax (emotion_out .numpy (),axis =-1 )[0 ]

        return {
        "text":text ,
        "sentiment":self .sentiment_map [sentiment_idx ],
        "emotion":self .emotion_map [emotion_idx ]
        }
