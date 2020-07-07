import csv, pyodbc


# set up some constants
MDB = '/home/tianqinl/Code/VisualConceptRouting/data_unzip/bedroom_train_lmdb/data.mdb'
DRV = '{Microsoft Access Driver (*.mdb)}'
PWD = 'pw'

# connect to db
con = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(DRV,MDB,PWD))
cur = con.cursor()

# run a query and get the results 
SQL = 'SELECT * FROM mytable;' # your query goes here
rows = cur.execute(SQL).fetchall()
cur.close()
con.close()

# you could change the mode from 'w' to 'a' (append) for any subsequent queries
with open('mytable.csv', 'w') as fou:
    csv_writer = csv.writer(fou) # default field-delimiter is ","
    csv_writer.writerows(rows)