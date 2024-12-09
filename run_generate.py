from numpy.lib.npyio import savetxt
import datetime
import os

from sympy.physics.units import current

import gen_net
from net_gen import conn_random_er, matrix_save

data_folder = 'Data'
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

current_folder = data_folder + '/' + str(datetime.datetime.now())
if not os.path.isdir(current_folder):
    os.makedirs(current_folder)

conn_file = current_folder + '/conn.txt'
conn_matrix = conn_random_er()
matrix_save(conn_matrix[0], conn_file)