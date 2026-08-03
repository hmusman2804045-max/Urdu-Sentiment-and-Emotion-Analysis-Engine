import os 
import sys 
import torch 
from transformers import AutoTokenizer ,AutoModelForSequenceClassification 


sys .stdout .reconfigure (encoding ='utf-8')

BASE_DIR =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
MAX_LENGTH =128 



WORD_START ="▁"


def get_word_attention (text ,model ,tokenizer ,max_length =MAX_LENGTH ):
    """
    Return which words the model 'looked at' most, as [{'word': str, 'score': float}].

    How it works:
      1. tokenize the text
      2. run the model asking for attention weights back
      3. take the LAST transformer layer (closest to the classification decision)
      4. average over all 12 attention heads
      5. keep the <s> (CLS) token's row -- how much <s> attends to every other token.
         <s> is the vector that actually feeds the classifier, so its attention is a
         rough proxy for "which tokens mattered".
      6. glue sub-word pieces back into whole words and scale scores to 0-1
    """
    inputs =tokenizer (text ,return_tensors ="pt",truncation =True ,max_length =max_length )

    model .eval ()
    with torch .no_grad ():
        outputs =model (**inputs ,output_attentions =True )


    last_layer =outputs .attentions [-1 ]
    avg_heads =last_layer .mean (dim =1 )[0 ]
    cls_attention =avg_heads [0 ]

    tokens =tokenizer .convert_ids_to_tokens (inputs ["input_ids"][0 ])
    special_tokens =set (tokenizer .all_special_tokens )

    words =[]
    scores =[]

    for token ,score in zip (tokens ,cls_attention .tolist ()):
        if token in special_tokens :
            continue 

        if token .startswith (WORD_START )or not words :

            words .append (token .replace (WORD_START ,""))
            scores .append (score )
        else :


            words [-1 ]+=token 
            scores [-1 ]=max (scores [-1 ],score )


    pairs =[(w ,s )for w ,s in zip (words ,scores )if w .strip ()]
    if not pairs :
        return []

    words =[w for w ,_ in pairs ]
    scores =[s for _ ,s in pairs ]


    lowest =min (scores )
    highest =max (scores )
    spread =highest -lowest 
    if spread ==0 :
        normalized =[1.0 for _ in scores ]
    else :
        normalized =[(s -lowest )/spread for s in scores ]

    return [{"word":w ,"score":round (s ,4 )}for w ,s in zip (words ,normalized )]


def load_model (model_dir ):
    """Load a saved model + tokenizer. eager attention is required to get attentions back."""
    tokenizer =AutoTokenizer .from_pretrained (model_dir )
    model =AutoModelForSequenceClassification .from_pretrained (
    model_dir ,attn_implementation ="eager")
    return model ,tokenizer 









SENTIMENT_LABELS =["Positive","Negative","Neutral"]

SAMPLES =[
"yeh film boht achi thi maza aa gaya",
"mujhe is service se sakht nafrat hai",
"aaj mausam theek hai kuch khaas nahi",
"main is khabar se boht dukhi hoon",
]


def main ():
    print ("="*60 )
    print ("  Phase 5: Attention Visualization Test")
    print ("="*60 )

    print ("\n[1/2] Loading sentiment model...")
    model_dir =os .path .join (BASE_DIR ,'models','sentiment_model')
    model ,tokenizer =load_model (model_dir )

    print ("\n[2/2] Scoring sample sentences...\n")
    for i ,text in enumerate (SAMPLES ,start =1 ):

        with torch .no_grad ():
            inputs =tokenizer (text ,return_tensors ="pt",truncation =True ,max_length =MAX_LENGTH )
            pred =model (**inputs ).logits .argmax (dim =-1 ).item ()

        word_scores =get_word_attention (text ,model ,tokenizer )

        print ("-"*60 )
        print (f"[{i }] {text }")
        print (f"    Predicted sentiment: {SENTIMENT_LABELS [pred ]}")
        for item in word_scores :
            bar ="#"*int (item ['score']*30 )
            print (f"    {item ['word']:<15} {item ['score']:.4f}  {bar }")

        top =sorted (word_scores ,key =lambda x :x ['score'],reverse =True )[:3 ]
        print ("    Top words: "+", ".join (f"{t ['word']} ({t ['score']:.2f})"for t in top ))

    print ("-"*60 )


if __name__ =="__main__":
    main ()
