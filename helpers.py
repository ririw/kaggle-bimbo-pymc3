import pandas
import numpy as np
import sqlite3
import csv

def prepare_database():
    con = sqlite3.connect('./data.sqlite3')
    con.execute('''
        create table data (
            id INTEGER,
            week_num INTEGER,
            sales_depo INTEGER,
            sales_channel INTEGER,
            route_id INTEGER,
            client_id INTEGER,
            product_id INTEGER,
            week_sales REAL,
            week_returns INTEGER,
            next_week_returns INTEGER,
            next_week_returns_peso REAL,
            adjusted_demand INTEGER,
            rand INTEGER
        )
        ''')
    with open('./train.csv') as f:
        reader = csv.reader(f)
        reader_iter = iter(reader)
        # Discard header
        next(reader_iter)
        for row in reader_iter:
            embiggened_row = [None] + row + [np.random.randint(100)]
            row_items = ','.join(['?'] * len(embiggened_row))
            con.execute('insert into data values (%s)' % row_items, embiggened_row)
    con.commit()
    with open('./test.csv') as f:
        reader = csv.reader(f)
        reader_iter = iter(reader)
        # Discard header
        next(reader_iter)
        for row in reader_iter:
            embiggened_row = row + ([None] * 5) + [np.random.randint(100)]
            row_items = ','.join(['?'] * len(embiggened_row))
            con.execute('insert into data values (%s)' % row_items, embiggened_row)
    con.commit()
    con.close()


prepare_database()
