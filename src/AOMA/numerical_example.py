import koma.oma
import numpy as np
import matplotlib.pyplot as plt
import koma.clustering
import geneate_vibration_data
import strid
import copy

# Main script for numerical example

path = "/../../data/vibration_data/data_stochastic_3_floor_"

number_of_realizations = 100

# Data generation

for i in range(number_of_realizations):
    geneate_vibration_data.generate_data(9, path + str(i) + ".npz")

# Define SSI parameters
i = 50
s = 1

orders = np.arange(1, 50, 1)
stabcrit = {'freq': 0.2, 'damping': 0.2, 'mac': 0.5 }

# Cov-SSI call and pole clustering
freq_modes = []

for j in range(number_of_realizations):

    data = np.load(path + str(j) + ".npz")
    y = data["y"]
    #print(y.shape)
    fs = data["fs"]
    #print(fs)
    true_f = data["true_frequencies"].transpose()
    true_xi = data["true_damping"].transpose()
    true_modeshapes = data["true_modeshapes"].transpose()

    # Cov-SSI
    ssid = strid.CovarianceDrivenStochasticSID(y, fs) #, ix_references)
    modes = {}
    for order in orders:
        A, C, G, R0 = ssid.perform(order, i)
        modes[order] = strid.Mode.find_modes_from_ss(A, C, ssid.fs)

    # Sorting routine
    lambdas = []
    phis = []

    for order in modes.keys():
        modes_in_order = modes[order]
        lambdas_in_order = []
        phis_in_order = []
        for mode in modes_in_order:
            lambdas_in_order.append(mode.eigenvalue)
            phis_in_order.append(mode.eigenvector)
        lambdas.append(np.array(lambdas_in_order))
        phis.append(np.array(phis_in_order).transpose())

    lambd_stab, phi_stab, orders_stab, ix_stab = koma.oma.find_stable_poles(lambdas, phis, orders, s,
                            stabcrit=stabcrit, valid_range={'freq': [0, np.inf], 'damping':[0, 0.2]},
                            indicator='freq', return_both_conjugates=False)

    #Pole clustering
    pole_clusterer = koma.clustering.PoleClusterer(lambd_stab, phi_stab, orders_stab, min_cluster_size=25,
                                    min_samples=10, scaling={'mac':1.0, 'lambda_real':1.0, 'lambda_imag': 1.0})
    prob_threshold = 0.99   #probability of pole to belong to cluster,
                            # based on estimated "probability" density function
    args = pole_clusterer.postprocess(prob_threshold=prob_threshold, normalize_and_maxreal=True)

    xi_auto, omega_n_auto, phi_auto, order_auto, probs_auto, ixs_auto = koma.clustering.group_clusters(*args)

    xi_mean = np.array([np.mean(xi_i) for xi_i in xi_auto])
    fn_mean = np.array([np.mean(om_i) for om_i in omega_n_auto])/2/np.pi

    xi_std = np.array([np.std(xi_i) for xi_i in xi_auto])
    fn_std = np.array([np.std(om_i) for om_i in omega_n_auto])/2/np.pi

    freq_modes.append([freq for freq in fn_mean])


new_freqs = np.empty(shape=[number_of_realizations, len(true_f)], dtype=object)


def remove_and_return_min_distance(lst, reference):
    min_element = min(lst, key=lambda x: abs(x - reference), default="EMPTY")
    if min_element in lst and abs(min_element-reference)<0.03:
        lst.remove(min_element)
        return min_element
    else:
        return None

#preallocating vector for sorted frequencies
new_freqs = np.array(np.empty(shape=[number_of_realizations, len(true_f)], dtype=object), dtype=np.float)
candidates = copy.deepcopy(freq_modes)

for i in range(number_of_realizations):
    for j in range(len(true_f)):
        candidate = remove_and_return_min_distance(candidates[i], true_f[j])
        new_freqs[i, j] = candidate

freqs = []
num = []

for i in range(number_of_realizations):
    freqs.extend(freq_modes[i])
    num.extend(np.ones_like(freq_modes[i])*i)

#Plot

colors =['tab:blue', 'tab:orange', 'tab:green', 'red', 'cyan', 'purple', 'magenta', 'yellow', 'brown']

plt.figure(figsize=(9, 5), dpi=300)
plt.axhline(y = true_f[0], color = 'r', linestyle = '--', label="FEM modes")
plt.scatter(np.array(num), np.array(freqs), marker='.', color='grey')

for i in range(len(true_f)):
    if (i<len(true_f)):
        plt.axhline(y = true_f[i], color = 'r', linestyle = '--')
    plt.plot(np.arange(0, number_of_realizations), new_freqs[:,i], color=colors[i], linestyle="-", marker=".",
             label="Est. mode " + str(i+1))

plt.grid()
plt.legend(bbox_to_anchor = (1,1))
plt.xlabel("Realization #")
plt.ylabel("$f_n$ [Hz]")
#plt.ylim([0,1.05])
plt.xlim([0,100])
plt.savefig("num_example.jpg", bbox_inches='tight')
plt.show()

print(np.count_nonzero(~np.isnan(new_freqs), axis=0))

print(np.count_nonzero(~np.isnan(new_freqs)))
print(len(freqs))

f_mean = np.nanmean(new_freqs, axis=0)
print(f_mean)

print((true_f))

print(100*np.abs(true_f-f_mean)/true_f)