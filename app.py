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


if __name__ == '__main__':
    app.run()
