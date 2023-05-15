import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dataloader_halogaland.dataloader as dl
from dataloader_halogaland.plot import plotModeShape
from strid.utils import modal_assurance_criterion
import scipy as sp




class ModeTrace:
    def __init__(self, reference_modes: list[dl.Mode], numb_analysis,
                 simcrit = {'freq': 0.1, 'mac': 0.2}):

        self.reference_modes = {}
        for i, mode in enumerate(reference_modes):
            new_mode = {i:mode}
            self.reference_modes.update(new_mode)

        self.numb_analysis = numb_analysis
        self.mode_trace =  np.empty(shape=[len(self.reference_modes), self.numb_analysis], dtype=object)
        self.simcrit = simcrit
        self.mode_type = ['Horizontal', 'Vertical', 'Horizontal', 'Vertical', 'Vertical',
                          'Vertical', 'Horizontal', 'Vertical', 'Vertical', 'Horizontal',
                          'Vertical', 'Torsional', 'Vertical', 'Torsional', 'Horizontal',
                          'Vertical', 'Vertical', 'Vertical', 'Torsional', 'Vertical']

    def add_modes_from_period(self, candidate_modes: list[dl.Mode], period: int):
        candidate_modes = candidate_modes
        f_tol = self.simcrit['freq']
        mac_tol = self.simcrit['mac']

        for key, ref_mode in self.reference_modes.items():
            for candidate in candidate_modes:
                delta_f = np.abs(ref_mode.frequency - candidate.frequency)/\
                                    np.max([ref_mode.frequency, candidate.frequency])
                mac_1 = 1 - modal_assurance_criterion(ref_mode.mode_shape, candidate.mode_shape)
                if delta_f < f_tol and mac_1 < mac_tol:
                    if not isinstance(self.mode_trace[key, period], dl.Mode):
                        self.mode_trace[key, period] = candidate
                        candidate_modes.remove(candidate)
                    else:
                        competitor = self.mode_trace[key, period]
                        comp_delta_f = np.abs(ref_mode.frequency - competitor.frequency)/\
                                    np.max([ref_mode.frequency, competitor.frequency])
                        comp_mac_1 = 1 - modal_assurance_criterion(ref_mode.mode_shape, competitor.mode_shape)
                        if (delta_f < comp_delta_f and
                                mac_1 < comp_mac_1):
                            self.mode_trace[key, period] = candidate
                            candidate_modes.remove(candidate)
                            candidate_modes.append(competitor)

    def add_all_modes(self, all_modes):
        for i in range(len(all_modes)):
            self.add_modes_from_period(all_modes[i], period=i)

    def get_frequencies_from_trace(self, trace):
        traces = self.mode_trace[trace,:]

        freqs = []

        for i in range(len(traces)):
            if isinstance(traces[i], dl.Mode):
                freqs.append((i, traces[i].frequency))

        return freqs

    def get_damping_from_trace(self, trace):
        traces = self.mode_trace[trace,:]

        freqs = []

        for i in range(len(traces)):
            if isinstance(traces[i], dl.Mode):
                freqs.append((i, traces[i].damping))

        return freqs

    def plot_frequency_distribution(self):
        fig, axs = plt.subplots(5, 4, figsize=(20, 20), dpi=300)
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            freqs = self.get_frequencies_from_trace(i)
            ax.set_title('Mode ' + str(i + 1) + ' - ' + str(self.mode_type[i]))
            ax.set_xlabel('f [Hz]')
            if len(freqs) == 0:
                continue
            freqs_mode = np.array(list(zip(*freqs))[1])
            ax.hist(freqs_mode, 20, label='AOMA')
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
            #ax.vlines(self.reference_modes[i].frequency, 0, 250, color='red', linestyles='dashed', \
            # label='FEM')
            text = 'n = ' + str(len(freqs_mode)) + '\nr = ' + \
                   f"{(100*len(freqs_mode)/self.numb_analysis):.1f}" + '%'
            ax.text(0.72, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox= dict(boxstyle='round', facecolor='white'))
            #ax.legend(loc='upper left')
        fig.suptitle('Frequency distribution', fontsize=14, y=0.99)
        fig.tight_layout()

        return fig

    def plot_damping_distribution(self):
        fig, axs = plt.subplots(5, 4, figsize=(20, 20), dpi=300)
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            damp = self.get_damping_from_trace(i)
            ax.set_title('Mode ' + str(i + 1)+ ' - ' + str(self.mode_type[i]))
            ax.set_xlabel('Damping')
            if len(damp) == 0:
                continue
            damp_mode = np.array(list(zip(*damp))[1])
            ax.hist(damp_mode, 20)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
        fig.suptitle('Damping distribution', fontsize=20, y=0.99)
        fig.tight_layout()

        return fig

    def plot_freq_vs_temp_corr(self, temps):
        fig, axs = plt.subplots(5, 4, figsize=(20, 20), dpi=300)
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            freqs = self.get_frequencies_from_trace(i)
            ax.set_title('Mode ' + str(i + 1)+ ' - ' + str(self.mode_type[i]))
            ax.set_xlabel('f [Hz]')
            ax.set_ylabel('Temperature [$^\circ$C]')
            if len(freqs) == 0:
                continue
            indexes = np.array(list(zip(*freqs))[0])
            temps_for_mode = temps[indexes]
            freqs_mode = np.array(list(zip(*freqs))[1])

            # Find regression line
            b, a = np.polyfit(np.array(freqs_mode), np.array(temps_for_mode), deg=1)
            xseq = np.linspace(0, 10, num=len(temps_for_mode))

            # Pearson correlation coefficient
            corr = sp.stats.linregress(np.array(freqs_mode), np.array(temps_for_mode))

            ax.scatter(np.array(freqs_mode), np.array(temps_for_mode), alpha=0.7)
            ax.plot(xseq, a + b * xseq, color='red')
            ax.set_xlim([np.mean(freqs_mode) - np.std(freqs_mode) * 5,
                         np.mean(freqs_mode) + np.std(freqs_mode) * 5])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
            ax.set_ylim([-15, 10])
            text = 'r = ' + f"{corr.rvalue:.2f}"
            ax.text(0.72, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white'))

        fig.suptitle('Frequency vs temperature correlation', fontsize=20, y=0.99)
        fig.tight_layout()

        return fig

    def plot_damp_vs_wind_corr(self, wind):
        fig, axs = plt.subplots(5, 4, figsize=(20, 20), dpi=300)
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            damp = self.get_damping_from_trace(i)
            ax.set_title('Mode ' + str(i + 1)+ ' - ' + str(self.mode_type[i]))
            ax.set_xlabel('Damping')
            ax.set_ylabel('Mean wind speed [m/s]')
            if len(damp) < 1:
                continue
            indexes = np.array(list(zip(*damp))[0])
            wind_for_mode = wind[indexes]
            damp_mode = np.array(list(zip(*damp))[1])

            # Find regression line
            b, a = np.polyfit(np.array(damp_mode), np.array(wind_for_mode), deg=1)
            xseq = np.linspace(0, 10, num=len(wind_for_mode))

            # Pearson correlation coefficient
            corr = sp.stats.linregress(np.array(damp_mode), np.array(wind_for_mode))

            ax.scatter(np.array(damp_mode), np.array(wind_for_mode), alpha=0.7)
            ax.plot(xseq, a + b * xseq, color='red')
            ax.set_xlim([0, np.mean(damp_mode) + np.std(damp_mode) * 5])
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
            ax.set_ylim([0, 30])
            text = 'r = ' + f"{corr.rvalue:.2f}"
            ax.text(0.72, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white'))

        fig.suptitle('Damping vs wind correlation', fontsize=20, y=0.99)
        fig.tight_layout()

        return fig

