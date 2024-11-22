import os

import gen_net

for i in range(2, 16, 2):
    count = 0
    for _ in range(100):
        conn_net = gen_net.conn_small_ws(1000, i)
        if gen_net.is_strongly_connected(conn_net):
            count = count + 1
    print(i, count, count / 100)

os.system('play -nq -t alsa synth {} sine {}'.format(2, 440))  # 2 second at 440hz
