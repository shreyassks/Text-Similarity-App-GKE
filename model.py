import numpy as np
from sentence_transformers import SentenceTransformer
from laserembeddings import Laser
import spacy
import argparse
import logging

METHODS = {
    'LASER': {
        'class': "Laser_Multi",
        'file': None
    },
    'DISTILBERT': {
        'class': "Distilbert",
        'file': None
    },
    'BERT LARGE NLI': {
        'class': "BERTlarge",
        'file': None
    },

}
    
def tokenizer(text: str):
    "Tokenize input string using a spaCy pipeline"
    nlp = spacy.blank('en')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(text)
    tokenized_text = ' '.join(token.text for token in doc)
    return tokenized_text

def embedding_class(method: str):
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_()

#embedding_class('BERTLARGE')

def cosine_similarity(a, b):
 # a and b are N dim Vectors
     sim = np.dot(a,b) / np.multiply(np.linalg.norm(a),np.linalg.norm(b))
     return round(float(sim), 3)

class Laser_Multi:
    
    def __init__(self):
        self.model = Laser()
    
    def predict(self, sent1, sent2):
        laser_emb = self.model.embed_sentences([sent1, sent2], lang = 'en')
        similarity = cosine_similarity(laser_emb[0], laser_emb[1])
        return round(similarity,2)
    
#laser = Laser_Multi()
#print("Laser Prediction:-", str(laser.predict("I am fine", "Are you good")))

class Distilbert:
    
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased')
        
    def predict(self, sent1, sent2):
        emb_1 = self.model.encode(sent1)
        emb_2 = self.model.encode(sent2)
        similarity = cosine_similarity(emb_1[0], emb_2[0])
        
        return similarity
    
#distil = Distilbert()
#print("Distil BERT Prediction:-", str(distil.predict("I am fine", "Are you good")))

class BERTlarge:
    
    def __init__(self):
        self.model = SentenceTransformer('bert-large-nli-mean-tokens')
          
    def predict(self, sent1, sent2):
        emb_1 = self.model.encode(sent1)
        emb_2 = self.model.encode(sent2)
        similarity = cosine_similarity(emb_1[0], emb_2[0])
        return similarity 
    
#bert = BERTlarge()
#print("BERT Large Prediction:-", str(bert.predict("I am fine", "Are you good")))
    
def embedder(method: str, text1, text2):

    model = embedding_class(method)
    similarity = model.predict(text1, text2)
    pred = round(similarity, 2)
    logging.info("Model Prediction Successful")

    return str(pred*100) + ' %'

#embedder("BERTLARGE", "I am fine", "Are you good")

#embedder("LASER", "I am fine", "Are you good")

def main(samples):
    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, nargs='+', help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)
    args = parser.parse_args()

    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
            
        print("Method: {}".format(method.upper()))
        
        text1 = tokenizer(samples[0])
        text2 = tokenizer(samples[1])
        sim = embedder(method, text1, text2)
            
        print("Similarity Score -", str(sim * 100) + '%')    
        
if __name__ == "__main__":
    # Evaluation text
    samples = [
        "It 's not horrible , just horribly mediocre .",
        "The cast is uniformly excellent ... but the film itself is merely mildly charming .",
    ]
    main(samples)
















