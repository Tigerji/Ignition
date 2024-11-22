import math
import random

import matplotlib.pyplot as plt


def conn_random_er(num_neuron=10, num_conn=4):  # Erdos Renyl Model G(n, p)
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        targets.remove(i)  # no self_connection
        for j in targets:
            if random.random() <= num_conn / num_neuron:
                conn_net[i].append(j)
    return conn_net, 'Directed Erodos Renyl random connection, G(' + str(num_neuron) + ',' + str(
        num_conn / num_neuron) + ')'


def conn_random_fixout(num_neuron=10, num_conn=4):
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        targets.remove(i)  # no self_connection
        selected = random.sample(targets, num_conn)
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net, 'Fixed(' + str(num_conn) + ') post-synaptic random connection'


def conn_random_fixin(num_neuron=10, num_conn=4):
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        targets.remove(i)
        selected = random.sample(targets, num_conn)
        for j in selected:
            conn_net[j].append(i)
    return conn_net, 'Fixed(' + str(num_conn) + ') pre-synaptic random connection'


def conn_random_fixout_linear(num_neuron=10, num_conn=4):  # linearly distributed based on distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 if i == j else num_neuron / 2 - min(abs(i - j), num_neuron - abs(i - j)) for j in targets]
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net, 'Fixed(' + str(num_conn) + ') post-synaptic linear weighted random connection'


def conn_random_fixout_inverse(num_neuron=10, num_conn=4):  # weight is the inverse of distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 if j == i else 1 / min(abs(i - j), num_neuron - abs(i - j)) for j in targets]
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net, 'Fixed(' + str(num_conn) + ') post-synaptic inverse weighted random connection'


def conn_random_fixout_normal(num_neuron=10, num_conn=4, sigma=20):  # normal distributed based on distance
    conn_net = [[] for _ in range(num_neuron)]
    for i in range(num_neuron):
        targets = list(range(num_neuron))
        selected = []
        weight = [0 for _ in range(num_neuron)]
        for j in targets:
            if i != j:
                s = min(abs(i - j), num_neuron - abs(i - j)) / num_neuron * sigma
                weight[j] = 1 / (1 * math.sqrt(2 * math.pi)) * math.exp(-((s - 0) ** 2 / (2 * 1 ** 2)))
        for j in range(num_conn):
            new_target = random.choices(targets, weights=weight)
            weight[new_target[0]] = 0
            selected = selected + new_target
        selected.sort()
        conn_net[i].extend(selected)
    return conn_net, 'Fixed(' + str(num_conn) + ') post-synaptic normal weighted random connection'


def conn_small_ws(num_neuron=10, num_conn=4, mut_ratio=0.05):  # Watts-Strogatz model
    conn_net = [[] for _ in range(num_neuron)]
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws')
    else:
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)

            mut_target = list(range(num_neuron))
            del mut_target[i]  # no self connection
            for k in curr_net:
                mut_target.remove(k)  # no repetition

            for j in range(num_conn):  # mutation
                if random.random() < mut_ratio:
                    new_target = random.sample(mut_target, 1)
                    mut_target.remove(new_target[0])
                    curr_net[j] = new_target[0]
            conn_net[i] = sorted(curr_net)
    return conn_net, 'small_ws'


def conn_small_dist(num_neuron=10, num_conn=4, mut_ratio=0.05):  # Watts-Strogatz model
    conn_net = [[] for _ in range(num_neuron)]
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws')
    else:
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)

            mut_target = list(range(num_neuron))
            del mut_target[i]  # no self connection
            for k in curr_net:
                mut_target.remove(k)  # no repetition

            for j in range(num_conn):  # mutation
                if random.random() < mut_ratio:
                    weight = [num_neuron / 2 - min(abs(i - s), num_neuron - abs(i - s)) for s in mut_target]
                    new_target = random.choices(mut_target, weights=weight)
                    mut_target.remove(new_target[0])
                    curr_net[j] = new_target[0]
            conn_net[i] = sorted(curr_net)
    return conn_net, 'small_dist'


def conn_small_ws_rgr(num_neuron=10, num_conn=4, mut_ratio=0.05):  # Watts-Strogatz richer get richer
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws_rgr')
    else:
        conn_net = [[] for i in range(num_neuron)]
        stats_net = [num_conn for i in range(num_neuron)]
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)
            conn_net[i] = sorted(curr_net)

        # mutation
        shuffled_neuron = list(range(num_neuron))
        random.shuffle(shuffled_neuron)
        for i in shuffled_neuron:
            shuffled_conn = list(range(num_conn))
            random.shuffle(shuffled_conn)
            for j in shuffled_conn:
                mut_target = list(range(num_neuron))
                temp_stats = stats_net[:]
                temp_stats[i] = 0  # no self connection
                for k in conn_net[i]:
                    temp_stats[k] = 0  # no repetition
                if random.random() < mut_ratio:
                    new_target = random.choices(mut_target, temp_stats)  # rich get richer
                    temp_stats[new_target[0]] = 0
                    stats_net[conn_net[i][j]] = stats_net[conn_net[i][j]] - 1
                    stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                    conn_net[i][j] = new_target[0]
        return conn_net, 'small_ws_rgr'


def conn_small_ws_fit(num_neuron=10, num_conn=4, mut_ratio=0.05):  # Watts-Strogatz fitness
    if num_conn % 2 == 1 or num_conn * 2 >= num_neuron or num_neuron < num_conn:
        print('Parameter error in conn_small_ws_rgr')
    else:
        conn_net = [[] for i in range(num_neuron)]
        stats_net = [random.random() for i in range(num_neuron)]  # fitness stats
        for i in range(num_neuron):
            curr_net = []  # connected neurons for current pre-synaptic neuron
            for j in range(1, int(num_conn / 2) + 1):  # create ring
                if i - j >= 0:
                    curr_net.append(i - j)
                else:
                    curr_net.append(i - j + num_neuron)
                if i + j < num_neuron:
                    curr_net.append(i + j)
                else:
                    curr_net.append(i + j - num_neuron)
            conn_net[i] = sorted(curr_net)

        # mutation
        shuffled_neuron = list(range(num_neuron))
        random.shuffle(shuffled_neuron)
        for i in shuffled_neuron:
            shuffled_conn = list(range(num_conn))
            random.shuffle(shuffled_conn)
            for j in shuffled_conn:
                mut_target = list(range(num_neuron))
                temp_stats = stats_net[:]
                temp_stats[i] = 0  # no self connection
                for k in conn_net[i]:
                    temp_stats[k] = 0  # no repetition
                if random.random() < mut_ratio:
                    new_target = random.choices(mut_target, temp_stats)  # rich get richer
                    temp_stats[new_target[0]] = 0
                    conn_net[i][j] = new_target[0]
        return conn_net, 'small_ws_fit'


def conn_small_nw(num_neuron=10, num_conn=4, n_inout=1):  # n_inout based Newman–Watts model
    if num_conn < n_inout or num_conn < 2:
        print('Wrong parameters in conn_small_nw')
    else:
        conn_net = [[-1 for _ in range(n_inout)] for _ in range(num_neuron)]
        stats_net = [n_inout for _ in range(num_neuron)]
        neuron_list = list(range(num_neuron))
        for i in range(n_inout):
            shuffled_list = neuron_list[:]
            random.shuffle(shuffled_list)
            j = 0
            while j < num_neuron:
                if shuffled_list[j] in conn_net[j] or shuffled_list[j] == j:  # duplicate or self-connection
                    j = -1
                    random.shuffle(shuffled_list)
                j = j + 1
            for j in neuron_list:
                conn_net[j][i] = shuffled_list[j]

        for i in range(n_inout, num_conn):  # new connections
            random.shuffle(shuffled_list)
            for j in shuffled_list:  # random selection sequence
                temp_stats = stats_net[:]
                temp_stats[j] = 0  # no self connection
                for k in conn_net[j]:
                    temp_stats[k] = 0  # no duplicate
                new_target = random.choices(neuron_list, temp_stats)
                conn_net[j].append(new_target[0])
                stats_net[conn_net[j][i]] = stats_net[conn_net[j][i]] + 1

        return conn_net, 'small_nw'


def conn_scalefree_ba(num_neuron=10, num_conn=4):  # Barabási–Albert model as in (Prettejohn, et. al, 2011)
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ba')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net = [0 for _ in range(num_neuron)]
        for i in range(num_conn):
            neuron_list = list(range(num_conn))
            neuron_list.remove(i)
            conn_net[i].extend(neuron_list)
            stats_net[i] = stats_net[i] + num_conn
        for i in range(num_conn, num_neuron):
            neuron_list = list(range(i))
            temp_stats = stats_net[0:i]
            stats_net[i] = stats_net[i] + 1
            for j in range(int(num_conn / 2)):
                new_target = random.choices(neuron_list, temp_stats)
                conn_net[i].append(new_target[0])
                conn_net[new_target[0]].append(i)
                stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                stats_net[i] = stats_net[i] + 1
                temp_stats[new_target[0]] = 0
        return conn_net, 'scale-free Barabási–Albert'


def conn_scalefree_price(num_neuron=10, num_conn=4):
    # Scale free network Price model
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ba')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net = [0 for _ in range(num_neuron)]
        stats_net[0] = 1
        for i in range(1, num_neuron):
            stats_net[i] = stats_net[i] + 1
            neuron_list = list(range(i))
            if len(neuron_list) <= num_conn:  # connect to all previous neurons
                for j in neuron_list:
                    conn_net[i].append(j)
                    stats_net[j] = stats_net[j] + 1
            else:  # selection based on in-degree+1
                temp_stats = stats_net[0:i]
                for _ in range(num_conn):
                    new_target = random.choices(neuron_list, temp_stats)
                    conn_net[i].append(new_target[0])
                    stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                    temp_stats[new_target[0]] = 0
        conn_net[0].append(1)
        return conn_net, 'scale-free Price'


def conn_scalefree_chris(num_neuron=10, num_conn=4, n_inout=1):  # chris model without proper citation
    if num_conn < n_inout or num_conn < 2:
        print('Wrong parameters in conn_scalefree_chris')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net = [1 for _ in range(num_neuron)]
        neuron_list = list(range(num_neuron))
        for j in range(n_inout):  # num_conn * n_inout network
            shuffled_list = list(range(num_conn))
            random.shuffle(shuffled_list)
            j = 0
            while j < num_conn:
                if shuffled_list[j] in conn_net[j] or shuffled_list[j] == j:  # duplicate or self-connection
                    j = -1
                    random.shuffle(shuffled_list)
                j = j + 1
            for j in range(num_conn):
                conn_net[j].append(shuffled_list[j])
                stats_net[shuffled_list[j]] = stats_net[shuffled_list[j]] + 1

        for i in range(num_conn, num_neuron):  # new connections
            shuffled_list = neuron_list[0:i]
            temp_stats = stats_net[0:i]
            for j in range(num_conn):
                new_target = random.choices(shuffled_list, temp_stats)
                conn_net[i].append(new_target[0])
                stats_net[new_target[0]] = stats_net[new_target[0]] + 1
                temp_stats[new_target[0]] = 0
        return conn_net, 'scalefree_chris'


def conn_scalefree_ptc(num_neuron=10, num_conn=4, ratio=0.8, n_inout=1):  # Scale free network from (Pi, et al., 2021)
    if num_conn + 1 > num_neuron:
        print('Wrong parameters in conn_scalefree_ptc')
    else:
        conn_net = [[] for _ in range(num_neuron)]
        stats_net_in = [1 for _ in range(num_neuron)]
        stats_net_out = [1 for _ in range(num_neuron)]
        for j in range(n_inout):  # (num_conn+1) * n_inout network
            shuffled_list = list(range(num_conn + 1))
            random.shuffle(shuffled_list)
            j = 0
            while j < num_conn + 1:
                if shuffled_list[j] in conn_net[j] or shuffled_list[j] == j:  # duplicate or self-connection
                    j = -1
                    random.shuffle(shuffled_list)
                j = j + 1
            for j in range(num_conn + 1):
                conn_net[j].append(shuffled_list[j])
                stats_net_in[shuffled_list[j]] = stats_net_in[shuffled_list[j]] + 1
                stats_net_out[j] = stats_net_out[j] + 1
        for i in range(num_conn + 1, num_neuron):  # new connections
            neuron_list = list(range(i))
            for _ in range(num_conn):
                if random.random() < ratio:  # edges from new neuron
                    temp_stats_in = stats_net_in[0:i]
                    for j in conn_net[i]:  # remove duplicates
                        temp_stats_in[j] = 0
                    new_target = random.choices(neuron_list, temp_stats_in)
                    conn_net[i].append(new_target[0])
                    stats_net_in[new_target[0]] = stats_net_in[new_target[0]] + 1
                    stats_net_out[i] = stats_net_out[i] + 1
                else:  # edges from existing neurons
                    temp_stats_in = stats_net_in[0:i]
                    temp_stats_out = stats_net_out[0:i]
                    new_source = random.choices(neuron_list, temp_stats_out)
                    temp_stats_in[new_source[0]] = 0  # no self connection
                    for j in conn_net[new_source[0]]:  # remove duplicates
                        temp_stats_in[j] = 0
                    if sum(temp_stats_in) > 0:  # not fully connected neuron
                        new_target = random.choices(neuron_list, temp_stats_in)
                        conn_net[new_source[0]].append(new_target[0])
                        stats_net_in[new_target[0]] = stats_net_in[new_target[0]] + 1
                        stats_net_out[new_source[0]] = stats_net_out[new_source[0]] + 1
        return conn_net, 'scale-free PTC'


def is_strongly_connected(conn_net):
    num_nodes = len(conn_net)
    all_check = [False for _ in range(num_nodes)]
    for i in range(num_nodes):
        visited = [False for _ in range(num_nodes)]
        checking = [i]
        while checking:
            u = checking.pop()
            visited[u] = True
            for j in conn_net[u]:
                if not visited[j]:
                    checking.append(j)
        all_check[i] = all(visited)
    return all(all_check)



def demo_random(num_neuron=1000):
    import matplotlib.pyplot as plt
    weight_linear = [0 for _ in range(num_neuron)]
    weight_inverse = [0 for _ in range(num_neuron)]
    weight_normal = [0 for _ in range(num_neuron)]
    targets = list(range(num_neuron))
    i = int(num_neuron / 5)
    for j in targets:
        if i != j:
            weight_linear[j] = num_neuron / 2 - min(abs(i - j), num_neuron - abs(i - j))
            weight_inverse[j] = 1 / min(abs(i - j), num_neuron - abs(i - j))
            s = min(abs(i - j), num_neuron - abs(i - j)) / num_neuron * 20
            weight_normal[j] = 1 / (1 * math.sqrt(2 * math.pi)) * math.exp(-((s - 0) ** 2 / (2 * 1 ** 2)))
    plt.plot(weight_linear)
    plt.plot(weight_inverse)
    plt.plot(weight_normal)
    plt.show()


def shortest_length(conn_net):
    # get the average shortest path length for any two nodes in conn_net
    # use hash map larger amounts
    num_nodes = len(conn_net)
    aver_length = []
    max_length = []
    for i in range(num_nodes):
        lengths = [num_nodes + 1 for _ in range(num_nodes)]
        checking = [i]
        count = 0
        while checking:
            new_check = []
            for j in checking:
                if lengths[j] > count:
                    lengths[j] = count
                if conn_net[j]:
                    for s in conn_net[j]:
                        if lengths[s] > count + 1:
                            new_check.append(s)
            count = count + 1
            checking = new_check
        aver_length.append(sum(lengths) / num_nodes)
        max_length.append(max(lengths))
    return sum(aver_length) / num_nodes, max(max_length)


def del_png():
    # delete all .png files under directory
    import os
    file_directory = os.path.dirname(os.path.realpath(__file__))
    files_in_directory = os.listdir(file_directory)
    filtered_files = [file for file in files_in_directory if file.endswith(".png")]
    for file in filtered_files:
        path_to_file = os.path.join(file_directory, file)
        os.remove(path_to_file)


def del_txt():
    # delete all .txt files under directory
    import os
    file_directory = os.path.dirname(os.path.realpath(__file__))
    files_in_directory = os.listdir(file_directory)
    filtered_files = [file for file in files_in_directory if file.endswith(".txt")]
    for file in filtered_files:
        path_to_file = os.path.join(file_directory, file)
        os.remove(path_to_file)


def matrix_save(matrix, data_file="demo_matrix.txt"):
    # save in matrix form, check demofile5.txt for example
    with open(data_file, 'w+') as f:
        for i in matrix:
            for j in i:
                f.write(str(j))
            f.write('\n')


def net_save(matrix, data_file="demo_net.txt"):
    # save in matrix form, check demofile5.txt for example
    with open(data_file, 'w+') as f:
        for i in matrix:
            for j in range(len(i)):
                f.write(str(i[j]))
                if j < len(i) - 1:
                    f.write(',')
            f.write('\n')


def spike_save(spikes, data_file='demo_spike.txt'):
    # save spike data, check demospike5.txt for example
    with open(data_file, 'w+') as f:
        for i in range(len(spikes)):
            spike_number = []
            spike_timings = spikes[i]
            if len(spike_timings) > 1:
                for j in spike_timings:
                    spike_number.append("%.1f" % float(j))
            f.write(str(spike_number) + '\n')


def matrix_load(data_file='demo_matrix.txt'):
    # load in matrix form, check demofile5.txt for example
    with open(data_file, 'r') as f:
        data = []
        for i in f.readlines():
            removed = i[1:-2].replace('[', '')
            temp_matrix = []
            if len(removed) > 0:
                sub_matrix = removed.split(']')
                for j in sub_matrix:
                    temp_matrix.append([int(j.split(',')[0]), float(j.split(',')[1])])
            data.append(temp_matrix)
    return data


def spike_load(data_file='demo_spike.txt'):
    with open(data_file, 'r') as f:
        data = []
        for i in f.readlines():
            curr_line = []
            if len(i) > 3:
                for j in i[0:-2].split(','):
                    curr_line.append(float(j[2:-1]))
            data.append(curr_line)
    return data


def plot_binned(spike_data, sim_duration=1000):
    binned_fired = [[] for i in range(sim_duration)]
    for i in range(len(spike_data)):
        if spike_data[i]:
            for j in spike_data[i]:
                for k in range(-5, 5):
                    log_binned = math.ceil(j + k)
                    if 0 < log_binned < sim_duration and i not in binned_fired[log_binned]:
                        binned_fired[log_binned].append(i)
    binned_count = [len(binned_fired[i]) for i in range(sim_duration)]
    plt.plot(binned_count)
    plt.xlim(0, sim_duration)


def spike_gen(spike_condition, spike_series=[100, 120, 140, 160, 180, 200, 220, 240, 260, 280], num_spike=5):
    spike_time = [[] for i in range(num_spike)]
    if 'a' in spike_condition:
        spike_time[0] = spike_series
    if 'b' in spike_condition:
        spike_time[1] = spike_series
    if 'c' in spike_condition:
        spike_time[2] = spike_series
    if 'd' in spike_condition:
        spike_time[3] = spike_series
    if 'e' in spike_condition:
        spike_time[4] = spike_series
    return spike_time


def spike_gen_5(spike_condition=0, spike_series=[100, 120, 140, 160, 180, 200, 220, 240, 260, 280], num_spike=5):
    spike_time = [[]]
    if spike_condition == 0:
        spike_time = [[] for i in range(num_spike)]
        for i in range(num_spike):
            new_spike = spike_series[:]
            for j in range(len(new_spike)):
                new_spike[j] = new_spike[j] + i * 1000
            spike_time[i] = spike_time[i] + new_spike
    if spike_condition == 1:
        spike_time = [[] for i in range(num_spike)]
        for i in range(num_spike):
            new_spike = spike_series[:]
            for j in range(len(new_spike)):
                new_spike[j] = new_spike[j] + i * 1000
            if i == 0:  # ab
                spike_time[0] = spike_time[0] + new_spike
                spike_time[1] = spike_time[1] + new_spike
            if i == 1:  # bc
                spike_time[1] = spike_time[1] + new_spike
                spike_time[2] = spike_time[2] + new_spike
            if i == 2:  # ac
                spike_time[0] = spike_time[0] + new_spike
                spike_time[2] = spike_time[2] + new_spike
            if i == 3:  # ad
                spike_time[0] = spike_time[0] + new_spike
                spike_time[3] = spike_time[3] + new_spike
            if i == 4:  # ce
                spike_time[2] = spike_time[2] + new_spike
                spike_time[4] = spike_time[4] + new_spike

    if spike_condition == 2:
        spike_time = [[] for i in range(num_spike)]
        for i in range(3):
            new_spike = spike_series[:]
            for j in range(len(new_spike)):
                new_spike[j] = new_spike[j] + i * 1000
            if i == 0:  # a
                spike_time[0] = spike_time[0] + new_spike
            if i == 1:  # ab
                spike_time[0] = spike_time[0] + new_spike
                spike_time[1] = spike_time[1] + new_spike
            if i == 2:  # ad
                spike_time[0] = spike_time[0] + new_spike
                spike_time[3] = spike_time[3] + new_spike
    print(spike_time)
    return spike_time
