from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

class model_input(BaseModel):

    site : int
    hdlngth : float
    skullw : float
    totlngth : float
    taill : float
    footlgth : float
    earconch : float
    eye : float
    chest : float
    belly : float
    male : int
    female : int      
        
# loading the saved model
ml_model = pickle.load(open('ml_model_v01.sav', 'rb'))

@app.post('/age_prediction')
def age_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    site = input_dictionary['site']
    hdlngth = input_dictionary['hdlngth']
    skullw = input_dictionary['skullw']
    totlngth = input_dictionary['totlngth']
    taill = input_dictionary['taill']
    footlgth = input_dictionary['footlgth']
    earconch = input_dictionary['earconch']
    eye = input_dictionary['eye']
    chest = input_dictionary['chest']
    belly = input_dictionary['belly']
    male = input_dictionary['male']
    female = input_dictionary['female']

    input_list = [site, hdlngth, skullw, totlngth, taill, footlgth, earconch, eye, chest, belly, male, female]
    
    prediction = ml_model.predict([input_list])
    
    return prediction[0]
    