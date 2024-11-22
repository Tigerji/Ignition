import gen_net
import gen_check

conn_net = gen_net.conn_neuron_random(1000,10)

a = gen_check.stats_conn(conn_net)

print(conn_net)
print(a)