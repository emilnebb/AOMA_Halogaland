import numpy as np
import h5py
import matplotlib.pyplot as plt
import dataloader_halogaland.dataloader as dl
from dataloader_halogaland.plot import stabilization_diagram
import os
import koma.oma
import koma.clustering
import strid
from time import time
from datetime import datetime, timedelta
import warnings

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

analysis_length = 30  # minutes
cutoff_frequency = 1  # Hz
bridgedeck_only = False

loader = dl.HDF5_dataloader(os.getcwd()+'/../../../../../../../Volumes/LaCie/Halogaland_sixth_try.hdf5',
                            bridgedeck_only=bridgedeck_only)

output_path = os.getcwd() + '/results/output_AOMA_hangers.h5'

# Hyperparameters
i = 50  # number of block rows
s = 1
fs = 2
orders = np.arange(2, 200+2, 2)  # orders to perform system
stabcrit = {'freq': 0.2, 'damping': 0.2, 'mac': 0.5}  # Default
prob_threshold = 0.99
min_cluster_size = 50
min_samples = 20
scaling={'mac':1.0, 'lambda_real':1.0, 'lambda_imag': 1.0}

# Write hyperparameters as attributes to output file
with h5py.File(output_path, 'a') as hdf:
    hdf.attrs['i'] = i
    hdf.attrs['s'] = s
    hdf.attrs['order'] = np.max(orders)
    hdf.attrs['stabcrit_freq'] = stabcrit['freq']
    hdf.attrs['stabcrit_damping'] = stabcrit['damping']
    hdf.attrs['stabcrit_mac'] = stabcrit['mac']
    hdf.attrs['prob_threshold'] = prob_threshold
    hdf.attrs['min_cluster_size'] = min_cluster_size
    hdf.attrs['min_samples'] = min_samples
    hdf.attrs['scaling_mac'] = scaling['mac']
    hdf.attrs['scaling_lambda_real'] = scaling['lambda_real']
    hdf.attrs['scaling_lambda_imag'] = scaling['lambda_imag']
    hdf.attrs['analysis_length [min]'] = analysis_length
    hdf.attrs['cutoff_frequency'] = cutoff_frequency
    hdf.attrs['bridgedeck_only'] = bridgedeck_only

if bridgedeck_only:
    ix_references_y = (np.array([0, 2, 4, 6, 8, 10, 12, 14]) + 16)
    ix_references_z = (np.array([0, 2, 4, 6, 8, 10, 12, 14]) + 32)
    ix_references = np.concatenate((ix_references_y, ix_references_z)).tolist()
else:
    ix_references_y = (np.array([0, 2, 4, 6, 8, 10, 12, 14]) + 20)
    ix_references_z = (np.array([0, 2, 4, 6, 8, 10, 12, 14]) + 40)
    ix_references = np.concatenate((ix_references_y, ix_references_z)).tolist()


number_of_periods = len(loader.periods)
print("Number of periods to run " + str(number_of_periods))
number_in_sample = fs*60*analysis_length

skipped = 0
for period in range(number_of_periods-44):
    period = period + 44
    acc = loader.load_all_acceleration_data(loader.periods[period], preprosess=True,
                                            cutoff_frequency=cutoff_frequency, filter_order=10)

    # If all channels are present, proceed with split up in intervals and perform Cov-SSI and clustering,
    # if not, move to the next period
    if isinstance(acc, np.ndarray):
        acc = np.array_split(acc, acc.shape[0]/number_in_sample)
        #print(len(acc))

        for j in range(len(acc)):

            t0 = time()  # Start timer of computation process

            # Cov-SSI
            ssid = strid.CovarianceDrivenStochasticSID(acc[j].transpose(), fs, ix_references)
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

            lambd_stab, phi_stab, orders_stab, idx_stab = koma.oma.find_stable_poles(lambdas, phis, orders, s,
                                    stabcrit=stabcrit, valid_range={'freq': [0.05, np.inf], 'damping':[0, 0.2]},
                                    indicator='freq', return_both_conjugates=False)

            # HDBSCAN
            pole_clusterer = koma.clustering.PoleClusterer(lambd_stab, phi_stab, orders_stab,
                                                           min_cluster_size=min_cluster_size,
                                            min_samples=min_samples, scaling=scaling)

            args = pole_clusterer.postprocess(prob_threshold=prob_threshold, normalize_and_maxreal=True)

            xi_auto, omega_n_auto, phi_auto, order_auto, probs_auto, ixs_auto = koma.clustering.group_clusters(*args)

            xi_mean = np.array([np.median(xi_i) for xi_i in xi_auto])
            fn_mean = np.array([np.median(om_i) for om_i in omega_n_auto])/2/np.pi

            xi_std = np.array([np.std(xi_i) for xi_i in xi_auto])
            fn_std = np.array([np.std(om_i) for om_i in omega_n_auto])/2/np.pi

            # Sort and arrange modeshapes
            lambd_used, phi_used, order_stab_used, group_ixs, all_single_ix, probs = \
                pole_clusterer.postprocess(prob_threshold=prob_threshold)

            grouped_phis = koma.clustering.group_array(phi_used, group_ixs, axis=1)

            phi_extracted = np.zeros((len(grouped_phis), len(loader.acceleration_sensors)*3))

            for a in range(len(grouped_phis)):
                for b in range(np.shape(grouped_phis[a])[0]):
                   phi_extracted[a, b] = (np.real(np.median(grouped_phis[a][b])))



            # Load wind statistical data for analyzed time series
            mean_wind_speed, max_wind_speed, mean_wind_direction = loader.load_wind_stat_data(loader.periods[period],
                                                                                              analysis_length, j)
            mean_temp = loader.load_temp_stat_data(loader.periods[period], analysis_length, j)

            t1 = time()  # end timer of computation process
            print("Time serie " + str(j+1) + " of " + str(len(acc)) + " done in " + str(t1-t0) + " sec. Period " +
                  str(period+1) + " of " + str(number_of_periods) +
                  " done. Number of skipped periods: " + str(skipped)+".")

            # Prepare timestamp of the time series in process
            timestamp = (datetime.strptime(loader.periods[period], "%Y-%m-%d-%H-%M-%S") +
                         timedelta(minutes=j*analysis_length)).strftime("%Y-%m-%d-%H-%M-%S")

            # Save stabilization plot
            stab_diag = stabilization_diagram(acc[j], fs, 2, (np.array(omega_n_auto) / 2 / np.pi), np.array(order_auto),
                                              all_freqs=np.abs(lambd_stab) / 2 / np.pi, all_orders=orders_stab)
            plt.savefig("plots/stab_diag/hangers/stabilization_diagram_" + str(timestamp) + ".jpg")

            # Write results to h5 file
            with h5py.File(output_path, 'a') as hdf:
                G1 = hdf.create_group(timestamp)

                # Write results
                G1.create_dataset('Frequencies', data=fn_mean)
                G1.create_dataset('Damping', data=xi_mean)
                G1.create_dataset('Modeshape', data=phi_extracted)

                # Write attributes
                G1.attrs['Mean wind speed'] = mean_wind_speed
                G1.attrs['Max wind speed'] = max_wind_speed
                G1.attrs['Mean wind direction'] = mean_wind_direction
                G1.attrs['Mean temp'] = mean_temp
                G1.attrs['Execution time'] = (t1-t0)
                G1.attrs['Std of acceleration data'] = np.mean(np.std(acc[j], axis=0))

    else:
        skipped += 1
        print("One or more channels are missing, period skipped.")