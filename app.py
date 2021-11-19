from flask import Flask, request
import sqlite3
import os
import psycopg2
import pandas as pd
import json


DATABASE = 'database.db'
currentdirectory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

DB_HOST = 'ec2-44-198-154-255.compute-1.amazonaws.com'
DB_DATABASE = "d24cip913rl0bs"
DB_USER = 'ceotxzmvpuhqmt'
DB_PASSWORD = '3df8cf97cf7ad812fcef7c1a0f8ca6ed2c30ba2c8ec71d31636fa3ec1b3b9bb0'

# path = sys.path[0]


# API
@app.route('/')
def hello_world():
    # read in data

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

    # cur.execute(
    #     '''CREATE TABLE companies (
    #         name TEXT,
    #         industry TEXT,
    #         roa float,
    #         moneyfund float,
    #         totalretainedearningsratio float,
    #         fixedassets float,
    #         financialexpenses float,
    #         employeecompensationpayable float,
    #         quickratio float
    #          )'''
    # )
    # conn.commit()
    cur.execute(
        '''CREATE TABLE users (
            name TEXT PRIMARY KEY, 
            email TEXT,
            password TEXT,
            bank TEXT,
            field TEXT
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
        '''INSERT INTO users VALUES(
            '{}','{}','{}','{}','{}','{}','{}','{}','{}'
             )'''.format(request.form['name'], request.form['industry'], request.form['roa'],
                         request.form['moneyfund'], request.form['totalretainedearningsratio'],
                         request.form['fixedassets'],
                         request.form['financialexpenses'], request.form['employeecompensationpayable'],
                         request.form['quickratio'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return 'Success', 200


@app.route('/user', methods=["POST"])
def set_user():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        '''INSERT INTO users VALUES(
            '{}','{}','{}','{}','{}'
             )'''.format(request.form['name'], request.form['email'], request.form['password'],
                         request.form['bank'], request.form['field'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return 'Success', 200

@app.route('/user', methods=["GET"])
def get_user():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE name='{}'".format(request.args.get('name'))
    )

    try:
        record = cur.fetchall()[0]
        record = {
            'name': record[0],
            'email': record[1],
            'password': record[2],
            'bank': record[3],
            'field': record[4]
        }
        response = json.dumps(record)

        cur.close()
        conn.close()
        return response, 200
    except:
        return 'no such user', 404


@app.route('/user', methods=["PUT"])
def edit_user():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        "UPDATE users SET email='{}', password='{}', bank='{}', field='{}' WHERE name='{}'"
        .format(request.form['email'], request.form['password'],
                request.form['bank'], request.form['field'], request.form['name'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return 'Success', 200

@app.route('/company', methods=["GET"])
def get_company():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM f_all WHERE id='{}'".format(request.args.get('id'))
    )
    record = cur.fetchall()[0]
    list = "id,ITM_1010,ITM_1020,ITM_1030,ITM_1040,ITM_1050,ITM_1060,ITM_1070,ITM_1080,ITM_1090,ITM_1100,ITM_1110,ITM_1120,ITM_1130,ITM_1140,ITM_1150,ITM_1160,ITM_1170,ITM_1180,ITM_1190,ITM_1200,ITM_1210,ITM_1230,ITM_1240,ITM_1250,ITM_1270,ITM_1280,ITM_1290,ITM_1300,ITM_1310,ITM_1320,ITM_1330,ITM_1340,ITM_1350,ITM_1360,ITM_1370,ITM_1380,ITM_1400,ITM_1401,ITM_1410,ITM_1420,ITM_1430,ITM_1450,ITM_1460,ITM_1480,ITM_1490,ITM_1500,ITM_1510,ITM_1520,ITM_1530,ITM_1540,ITM_1550,ITM_1560,ITM_1570,ITM_1580,ITM_1590,ITM_1600,ITM_1610,ITM_1620,ITM_1680,ITM_1710,ITM_1720,ITM_1730,ITM_1740,ITM_1750,ITM_1760,ITM_1780,ITM_1790,ITM_1820,ITM_1840,ITM_1850,ITM_1860,ITM_1870,ITM_1880,ITM_1890,ITM_1900,ITM_1920,ITM_1220,ITM_1260,ITM_1390,ITM_1440,ITM_1470,ITM_1640,ITM_1650,ITM_1660,ITM_1670,ITM_1690,ITM_1700,ITM_3010,ITM_3030,ITM_3040,ITM_3050,ITM_3060,ITM_3070,ITM_3080,ITM_3090,ITM_3100,ITM_3110,ITM_3130,ITM_3140,ITM_3150,ITM_3170,ITM_3180,ITM_3190,ITM_3200,ITM_3220,ITM_3230,ITM_3240,ITM_3260,ITM_3270,ITM_3280,ITM_3290,ITM_3300,ITM_3310,ITM_3320,ITM_3330,ITM_3340,ITM_3350,ITM_3360,ITM_3020,ITM_3160,ITM_3210,ITM_3370,ITM_3380,ITM_2050,ITM_2070,ITM_2140,ITM_2170,ITM_2172,ITM_2173,ITM_2010,ITM_2020,ITM_2030,ITM_2040,ITM_2060,ITM_2080,ITM_2090,ITM_2100,ITM_2110,ITM_2120,ITM_2130,ITM_2150,ITM_2160,ITM_2180,ITM_2190,ITM_2200,ITM_2210,ITM_2220,ITM_2250,ITM_2260,ITM_2270,ITM_2280,ITM_2290,ITM_2300,ITM_2320,ITM_2330,y,to_pay_exp,inventory_vs_expense,sales_profit,gross_profit,roa,profit_net_asset,liab_ratio,liab_equity,tangible_asset_liab,secure_liab,liquidity,subsidiary,industry_code,industry_name,size,enterprise_ownership,field".split(',')

    res = {}
    i = 0
    for item in list:
        res[item] = record[i]
        i += 1

    response = json.dumps(res)

    cur.close()
    conn.close()
    return response, 200

if __name__ == '__main__':
    app.run()
