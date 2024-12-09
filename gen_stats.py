# total_num(conn_net): return number of elements in conn_net
# stats_conn(conn_net): return stats_net, incoming connections to each neuron
# is_strongly_connected(conn_net): checking connectivity, return True/False
# shortest_length(conn_net): get shortest path length for any two nodes in conn_net

# def clustering_coef_cg(conn_net):
# def small-wordedness(conn_net):
# def degree-distribution(conn_net):

# todo
# spike_analysis(conn_matrix, datafile):
# assembly_learning(connector):


def total_num(conn_net):
    # number of elements in conn_net
    count = 0
    for i in conn_net:
        count = count + len(i)
    return count


def stats_conn(conn_net):
    # return the total incoming connections
    stats_net = [0 for _ in range(len(conn_net))]
    for i in range(len(conn_net)):
        if conn_net[i]:
            for j in conn_net[i]:
                stats_net[j] = stats_net[j] + 1
    return stats_net


def plot_conn_net(conn_net, file='demo', title='demo'):
    import matplotlib.pyplot as plt
    from datetime import datetime
    plt.cla()
    num_neuron = len(conn_net)
    for i in range(num_neuron):
        if conn_net[i]:
            x = [i for _ in conn_net[i]]
            plt.plot(x, conn_net[i], 'b,')
    plt.xlim(0, num_neuron)
    plt.ylim(0, num_neuron)
    plt.xlabel('Pre-synaptic Neurons')
    plt.ylabel('Post-synaptic Neurons')
    plt.title(title)
    plt.savefig(file + '_' + str(datetime.now()) + '.png')


def degree_dist(networks, file='demo', mode='normal'):
    degree_in = [0 for _ in networks]
    degree_out = []
    for i in networks:
        degree_out.append(len(i))
        for j in i:
            degree_in[j] = degree_in[j] + 1
    import matplotlib.pyplot as plt
    from datetime import datetime
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(degree_in)
    ax1.set_title('Pre-synaptic connections')
    ax1.set_ylabel('Number of connections')
    ax2 = fig.add_subplot(222)
    ax2.plot(degree_out)
    ax2.set_title('Post-synaptic connections')
    ax2.set_ylabel('Number of connections')
    ax3 = fig.add_subplot(223)
    ax3.hist(degree_in)
    ax3.set_title('Pre-synaptic degree distribution')
    ax3.set_ylabel('Number of neurons')
    if mode == 'log':
        ax3.set_yscale('log')
        ax3.set_xscale('log')
    ax4 = fig.add_subplot(224)
    ax4.hist(degree_out)
    ax4.set_title('Post-synaptic degree distribution')
    ax4.set_ylabel('Number of neurons')
    if mode == 'log':
        ax4.set_yscale('log')
        ax4.set_xscale('log')
    plt.tight_layout()
    plt.savefig(file + '_' + str(datetime.now()) + '.png')


network_file = 'sample_network.txt'
