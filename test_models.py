import os 
import sys 
import torch 
import numpy as np 
from transformers import AutoTokenizer ,AutoModelForSequenceClassification 

sys .stdout .reconfigure (encoding ='utf-8')

def main ():
    print ("="*60 )
    print ("  Loading Urdu Sentiment & Emotion Models...")
    print ("="*60 )

    base_dir =os .path .dirname (os .path .abspath (__file__ ))
    sentiment_dir =os .path .join (base_dir ,"models","sentiment_model")
    emotion_dir =os .path .join (base_dir ,"models","emotion_model")



    sentiment_map ={0 :"Positive 😊",1 :"Negative 😠",2 :"Neutral 😐"}
    emotion_map ={0 :"Joy 😄",1 :"Anger 😡",2 :"Fear 😨",3 :"Sadness 😢"}


    try :
        print ("Loading Tokenizer...")
        tokenizer =AutoTokenizer .from_pretrained (sentiment_dir )

        print ("Loading Sentiment Model...")
        sentiment_model =AutoModelForSequenceClassification .from_pretrained (sentiment_dir )

        print ("Loading Emotion Model...")
        emotion_model =AutoModelForSequenceClassification .from_pretrained (emotion_dir )
    except Exception as e :
        print (f"Error loading models. Are you sure they finished training? ({e })")
        return 

    print ("\nModels loaded successfully!")
    print ("Type an Urdu sentence (Roman or Script) to test them. Type 'exit' to quit.\n")


    while True :
        text =input ("Enter Urdu text: ")
        if text .strip ().lower ()in ['exit','quit','q']:
            break 
        if not text .strip ():
            continue 


        inputs =tokenizer (text ,return_tensors ="pt",truncation =True ,max_length =128 )


        with torch .no_grad ():
            sentiment_out =sentiment_model (**inputs ).logits 
            emotion_out =emotion_model (**inputs ).logits 


        sentiment_idx =np .argmax (sentiment_out .numpy (),axis =-1 )[0 ]
        emotion_idx =np .argmax (emotion_out .numpy (),axis =-1 )[0 ]

        print ("-"*40 )
        print (f"Sentiment : {sentiment_map [sentiment_idx ]}")
        print (f"Emotion   : {emotion_map [emotion_idx ]}")
        print ("-"*40 +"\n")

if __name__ =="__main__":
    main ()
