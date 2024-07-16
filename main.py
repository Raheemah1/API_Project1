from fastapi import FastAPI, HTTPException
from typing import Dict, List, Tuple
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

with open('rfr_model.pk1', 'rb') as file:
    model = pickle.load(file)

model_cols = ['age', 'bmi', 'children', 'region_northeast', 'region_northwest',
              'region_southeast', 'region_southwest', 'smoker_no', 'smoker_yes',
              'sex_female', 'sex_male']

def make_list()->Tuple[List, List]:
    cat_cols = [['yes', 'southwest', 'male'],
                ['no', 'southeast', 'female'],
                ['yes', 'northwest', 'male'],
                ['no', 'northeast', 'female']]

    num_cols = [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]
    return cat_cols, num_cols



# create an instance of our application
app = FastAPI(title = 'My first ML-API')

class ConverterInput(BaseModel):
    base_currency: float
    rate: float
    your_currency: str

class Columns(BaseModel):
    smoker: str
    region: str
    sex: str
    age: int
    children:int
    bmi:float


@app.get('/')
async def greet(name: str) -> Dict:
    """
    This endpoint greets a user. it does that by accepting their name and greeting them accordingly.

    name[string]: The name of the person to be greeted
    return [Dictionary]: The greetings as a dictionary
    """
    if name is int:
        raise ValueError("Does not accept integer")

    greeting = f'Hello {name} how are you?'
    return {'greeting': greeting}


@app.post('/convert_currency/')
async def converter(params: ConverterInput) -> Dict:
    """
    This endpoint converts any currency based on the exchange rate
    :param base_currency: The amount of the currency you are converting
    :param rate: The current exchange rate
    :param your_currency: The name of the currency you are converting to
    :return: A dictionary output of the amount in your currency
    """
    if params.rate < 0:
        raise HTTPException(status_code = 400, detail = "You cannot have a negative rate")
    else:
        amount = params.base_currency * params.rate
    return {'output':f'{amount} {params.your_currency}'}


@app.post('/predict/')
async def predict(columns: Columns):
    """

    :param columns:
    :return:
    """
    columns = dict(columns)
    columns = {k:v for k, v in columns.items()}
    cat_cols, num_cols = make_list()
    numerical = [columns['age'], columns['children'], columns['bmi']]
    categorical = [columns['smoker'], columns['region'], columns['sex']]
    cat_cols.append(categorical)
    num_cols.append(numerical)


    cat_cols = pd.DataFrame(cat_cols, columns=['smoker', 'region', 'sex'])
    cat_cols = pd.get_dummies(cat_cols, dtype=float)

    num_cols = pd.DataFrame(num_cols, columns=['age', 'children', 'bmi'])
    df = pd.concat([num_cols, cat_cols], axis=1)

    scaler = StandardScaler()
    cols = df.columns
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=cols)

    pred_df = df.drop([0,1,2,3], axis = 0)
    pred_df = pred_df[model_cols]
    prediction = model.predict(pred_df)

    return {'prediction': str(round(prediction[0],2))}

@app.post('/test')
async def test(columns:Columns):
    columns = dict(columns)
    columns = {k: v for k, v in columns.items()}
    return columns







