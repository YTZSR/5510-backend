from flask import Flask, request
import sqlite3
import os
import psycopg2
import pandas as pd
import sys
from model import feature_engineering_train_test_split, LR_model_train_save, XGB_classifier_train_save, Catboost_classifier_train_save, LightGBM_classifier_train_save, Stacking_model_train_valid_save

DATABASE = 'database.db'
currentdirectory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

DB_HOST = 'ec2-44-198-154-255.compute-1.amazonaws.com'
DB_DATABASE = "d24cip913rl0bs"
DB_USER = 'ceotxzmvpuhqmt'
DB_PASSWORD = '3df8cf97cf7ad812fcef7c1a0f8ca6ed2c30ba2c8ec71d31636fa3ec1b3b9bb0'


# API
@app.route('/')
def hello_world():
    # read in data
    print(sys.path)
    final_all = pd.read_csv(sys.path[0] + "/final_all.csv")
    col = final_all.columns
    print(col)

    return 'Hello World!'

# @app.route('/init', methods=["POST"])
# def init():
#     conn = sqlite3.connect('database.db')
#
#     conn.execute(
#         '''CREATE TABLE companies (
#             name TEXT,
#             industry TEXT,
#             roa double,
#             moneyfund double,
#             totalretainedearningsratio double,
#             fixedassets double,
#             financialexpenses double,
#             employeecompensationpayable double,
#             quickratio double
#              )'''
#     )
#     conn.commit()
#     conn.close()
#     return 'Success', 200


@app.route('/init_remote', methods=["POST"])
def init_remote():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        '''CREATE TABLE companies (
            name TEXT, 
            industry TEXT,
            roa float, 
            moneyfund float, 
            totalretainedearningsratio float,
            fixedassets float,
            financialexpenses float,
            employeecompensationpayable float,
            quickratio float
             )'''
    )
    conn.commit()
    cur.close()
    return 'Success', 200


@app.route('/drop', methods=["GET"])
def drop():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(
        '''DROP TABLE companies'''
    )

    cur.close()
    conn.close()

    return 'Success', 200

@app.route('/set_company', methods=["POST"])
def set_company():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        '''INSERT INTO companies VALUES(
            '{}','{}','{}','{}','{}','{}','{}','{}','{}'
             )'''.format(request.form['name'], request.form['industry'], request.form['roa'],
                         request.form['moneyfund'], request.form['totalretainedearningsratio'], request.form['fixedassets'],
                         request.form['financialexpenses'], request.form['employeecompensationpayable'], request.form['quickratio'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return 'Success', 200

if __name__ == '__main__':
    # Train model
    path = sys.path[0]
    final_all = pd.read_csv(path + "/final_all.csv")
    X_smo, X_test, y_smo, y_test, col_lst = feature_engineering_train_test_split(final_all)

    y_pred_lr, y_prob_lr = LR_model_train_save(path, X_smo, y_smo, X_test, y_test)
    res_prob, res_xgb, importance_xgb_lst = XGB_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
    pred_catboost, pred_cat_prob, imp_cat = Catboost_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
    y_pred_lgb, y_pred_dummy_lgb, imp_lgb = LightGBM_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
    pred_catboost2, pred_prob_catboost2, report = Stacking_model_train_valid_save(path, X_smo, y_smo, X_test, y_test,
                                                                                  col_lst)


    app.run()
