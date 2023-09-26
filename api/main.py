import os
import sqlite3
import pickle
import sys

import pandas as pd
import uvicorn
from fastapi import FastAPI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts import get_binary_data

app = FastAPI()

folder_path = os.path.abspath('..')

connector = sqlite3.connect(folder_path + '/processed_credit_risk.db')

clf_model = pickle.load(
    open(folder_path + '/deploy/person_default_on_file.pkl', 'rb')
)
reg_model = pickle.load(open(folder_path + '/deploy/loan_int_rate.pkl', 'rb'))


def format_data(df: pd.DataFrame) -> pd.DataFrame:
    letters = ['B', 'C', 'D', 'E', 'F', 'G']

    for letter in letters:
        if df['loan_grade'][0] == letter:
            df[f"loan_grade_{letter}"] = 1
            for l in letters:
                if l != letter:
                    df[f"loan_grade_{l}"] = 0

    df = df.drop(['loan_grade'], axis=1)

    home_ownerships = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']

    for home_ownership in home_ownerships:
        if df['person_home_ownership'][0] == home_ownership:
            df[f"person_home_ownership_{home_ownership}"] = 1
            for h_o in home_ownerships:
                if h_o != home_ownership:
                    df[f"person_home_ownership_{h_o}"] = 0

    df = df.drop(['person_home_ownership'], axis=1)

    loan_intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
                    'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']

    for loan_intent in loan_intents:
        if df['loan_intent'][0] == loan_intent:
            df[f"loan_intent_{loan_intent}"] = 1
            for l_i in loan_intents:
                if l_i != loan_intent:
                    df[f"loan_intent_{l_i}"] = 0

    df = df.drop(['loan_intent'], axis=1)

    return df


def get_result_from_db(client_id: int):
    cursor = connector.cursor()
    cursor.execute(
        "SELECT * FROM processed_credit_risk WHERE id=?", (client_id,)
    )
    result = cursor.fetchall()
    cursor.close()

    return result


def format_tuple_to_dict(data: tuple):
    tuple_keys = (
        'id',
        'person_age',
        'person_income',
        'person_home_ownership',
        'person_emp_length',
        'loan_intent',
        'loan_grade',
        'loan_amnt',
        'loan_int_rate',
        'loan_status',
        'loan_percent_income',
        'cb_person_default_on_file',
        'cb_person_cred_hist_length'
    )
    if len(tuple_keys) == len(data):
        data_dict = dict(zip(tuple_keys, data))
        return data_dict
    else:
        return {}


@app.get('/classifier/{client_id}')
async def get_client_classification(client_id: int):
    result = get_result_from_db(client_id=client_id)
    if result:
        data_dict = format_tuple_to_dict(result[0])
        df = pd.DataFrame(data_dict, index=[0])
        df_feat = format_data(df)
        df_feat = df_feat[['loan_int_rate', 'loan_status', 'loan_grade_C',
                           'loan_grade_D', 'loan_grade_E']]
        clf = clf_model.predict(df_feat)
        df_feat['predict_person_default_in_file'] = clf
        return df_feat.to_dict(orient='records')
    else:
        return "Result not found"


@app.get('/regressor/{client_id}')
async def get_client_regression(client_id: int):
    result = get_result_from_db(client_id=client_id)
    if result:
        data_dict = format_tuple_to_dict(result[0])
        df = pd.DataFrame(data_dict, index=[0])
        df_feat = format_data(df)
        df_feat['cb_person_default_on_file'] = df.apply(
            lambda x: get_binary_data(x['cb_person_default_on_file']), axis=1)
        df_feat = df_feat[['person_age', 'person_income', 'person_emp_length',
                           'loan_amnt', 'loan_status',
                           'cb_person_default_on_file',
                           'cb_person_cred_hist_length',
                           'person_home_ownership_RENT',
                           'loan_intent_EDUCATION',
                           'loan_intent_HOMEIMPROVEMENT',
                           'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
                           'loan_intent_VENTURE', 'loan_grade_B',
                           'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
                           'loan_grade_F', 'loan_grade_G']]
        reg = reg_model.predict(df_feat)
        df_feat['predict_loan_int_rate'] = reg
        return df_feat.to_dict(orient='records')
    else:
        return "Result not found"


if __name__ == '__main__':
    uvicorn.run('main:app', host="127.0.0.1", port=8000, reload=True)
