import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import src.AOMA.dataloader as dl
from strid.utils import modal_assurance_criterion
import scipy as sp
import koma.modal as modal




class ModeTrace:
    def __init__(self, reference_modes: list[dl.Mode], numb_analysis,
                 simcrit = {'freq': 0.4, 'mac': 0.5}):

        self.reference_modes = {}
        for i, mode in enumerate(reference_modes):
            new_mode = {i:mode}
            self.reference_modes.update(new_mode)

        self.numb_analysis = numb_analysis
        self.mode_trace = np.empty(shape=[len(self.reference_modes), self.numb_analysis], dtype=object)
        self.simcrit = simcrit
        self.mode_type = ['Horizontal', 'Vertical', 'Horizontal', 'Vertical', 'Vertical',
                          'Vertical', 'Horizontal', 'Cable', 'Vertical', 'Vertical', 'Horizontal',
                          'Vertical', 'Torsional', 'Cable', 'Cable', 'Vertical', 'Torsional', 'Horizontal',
                          'Vertical', 'Vertical', 'Vertical', 'Torsional', 'Vertical', 'Vertical']

    def add_modes_from_period(self, candidate_modes: list[dl.Mode], period: int):
        candidate_modes = candidate_modes
        f_tol = self.simcrit['freq']
        mac_tol = self.simcrit['mac']
        if len(candidate_modes[0].mode_shape) > 48:
            indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,40,
                   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]
        else:
            indexes = np.arange(0, 48, 1)

        for key, ref_mode in self.reference_modes.items():
            for candidate in candidate_modes:
                delta_f = np.abs(ref_mode.frequency - candidate.frequency)/\
                                    np.max([ref_mode.frequency, candidate.frequency])
                mac_1 = 1 - modal_assurance_criterion(ref_mode.mode_shape, candidate.mode_shape[indexes])
                if delta_f < f_tol and mac_1 < mac_tol:
                    if not isinstance(self.mode_trace[key, period], dl.Mode):
                        self.mode_trace[key, period] = candidate
                        candidate_modes.remove(candidate)
                    else:
                        competitor = self.mode_trace[key, period]
                        comp_delta_f = np.abs(ref_mode.frequency - competitor.frequency)/\
                                    np.max([ref_mode.frequency, competitor.frequency])
                        comp_mac_1 = 1 - modal_assurance_criterion(ref_mode.mode_shape, competitor.mode_shape[indexes])
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

        damps = []

        for i in range(len(traces)):
            if isinstance(traces[i], dl.Mode):
                damps.append((i, traces[i].damping))

        return damps

    def plot_frequency_distribution(self):
        fig, axs = plt.subplots(6, 4, figsize=(20, 20), dpi=300)
        axs = axs.ravel()
        remove = 0
        for i, ax in enumerate(axs):
            if i < len(self.reference_modes):
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
            else:
                remove += 1
                continue
        fig.suptitle('Frequency distribution', fontsize=14, y=0.99)
        fig.tight_layout()
        while remove > 0:
            fig.delaxes(axs[-remove])
            remove -= 1

        return fig

    def plot_damping_distribution(self):
        fig, axs = plt.subplots(6, 4, figsize=(20, 25), dpi=300)
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
        fig, axs = plt.subplots(6, 4, figsize=(20, 25), dpi=300)
        axs = axs.ravel()
        remove = 0
        for i, ax in enumerate(axs):
            if i < len(self.reference_modes):
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
        while remove > 0:
            fig.delaxes(axs[-remove])
            remove -= 1

        return fig

    def plot_damp_vs_wind_corr(self, wind):
        fig, axs = plt.subplots(6, 4, figsize=(20, 25), dpi=300)
        axs = axs.ravel()
        remove = 0
        for i, ax in enumerate(axs):
            if i < len(self.reference_modes):
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
        while remove > 0:
            fig.delaxes(axs[-remove])
            remove -= 1

        return fig

    def plot_AOMA_vs_FEM(self):

        fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=300)

        for i in range(len(self.reference_modes)):

            if self.mode_type[i] == 'Horizontal':
                axs[0].plot(self.reference_modes[i].frequency,
                            np.mean(np.array(self.get_frequencies_from_trace(i)[:])[:, 1]), marker='o', color='r')
                axs[0].plot([0, 1], [0, 1], transform=axs[0].transAxes, color='black')
                axs[0].set_xlabel('$f_{FEM}$ [Hz]')
                axs[0].set_ylabel('$f_{AOMA}$ [Hz]')
                axs[0].set_title(self.mode_type[i])
                axs[0].set_xticks(np.arange(0, 1.2, step=0.2))
                axs[0].set_yticks(np.arange(0, 1.2, step=0.2))
                axs[0].set_xlim([0, 1])
                axs[0].set_ylim([0, 1])

            elif self.mode_type[i] == 'Vertical':
                axs[1].plot(self.reference_modes[i].frequency,
                            np.mean(np.array(self.get_frequencies_from_trace(i)[:])[:, 1]), marker='o', color='r')
                axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, color='black')
                axs[1].set_xlabel('$f_{FEM}$ [Hz]')
                axs[1].set_ylabel('$f_{AOMA}$ [Hz]')
                axs[1].set_title(self.mode_type[i])
                axs[1].set_xticks(np.arange(0, 1.2, step=0.2))
                axs[1].set_yticks(np.arange(0, 1.2, step=0.2))
                axs[1].set_xlim([0, 1.1])
                axs[1].set_ylim([0, 1.1])

            elif self.mode_type[i] == 'Torsional':
                axs[2].plot(self.reference_modes[i].frequency,
                            np.mean(np.array(self.get_frequencies_from_trace(i)[:])[:, 1]), marker='o', color='r')
                axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color='black')
                axs[2].set_xlabel('$f_{FEM}$ [Hz]')
                axs[2].set_ylabel('$f_{AOMA}$ [Hz]')
                axs[2].set_title(self.mode_type[i])
                axs[2].set_xticks(np.arange(0, 1.2, step=0.2))
                axs[2].set_yticks(np.arange(0, 1.2, step=0.2))
                axs[2].set_xlim([0, 1])
                axs[2].set_ylim([0, 1])

            elif self.mode_type[i] == 'Cable':
                axs[3].plot(self.reference_modes[i].frequency,
                            np.mean(np.array(self.get_frequencies_from_trace(i)[:])[:, 1]), marker='o', color='r')
                axs[3].plot([0, 1], [0, 1], transform=axs[3].transAxes, color='black')
                axs[3].set_xlabel('$f_{FEM}$ [Hz]')
                axs[3].set_ylabel('$f_{AOMA}$ [Hz]')
                axs[3].set_title(self.mode_type[i])
                axs[3].set_xticks(np.arange(0, 1.2, step=0.2))
                axs[3].set_yticks(np.arange(0, 1.2, step=0.2))
                axs[3].set_xlim([0, 1])
                axs[3].set_ylim([0, 1])

        return fig


def plotModeShapeAOMA(tracer: ModeTrace, FEM_loader: dl.FEM_result_loader, type ='Vertical'):

    all_modeshapes = np.array(np.empty([tracer.mode_trace.shape[0], tracer.mode_trace.shape[1],
                                    len(tracer.mode_trace[0,0].mode_shape)]), dtype=np.float)

    for i in range(tracer.mode_trace.shape[0]):
        for j in range(tracer.mode_trace.shape[1]):
            if isinstance(tracer.mode_trace[i,j], dl.Mode):
                all_modeshapes[i, j, :] = tracer.mode_trace[i, j].mode_shape

    f_mean = []
    xi_mean = []
    ref_phi = np.zeros([48, len(tracer.reference_modes)])
    for i in range(len(tracer.reference_modes)):
        ref_phi[:, i] = tracer.reference_modes[i].mode_shape
        f_mean.append(np.mean(np.array(tracer.get_frequencies_from_trace(i)[:])[:, 1]))
        xi_mean.append(100*np.mean(np.array(tracer.get_damping_from_trace(i)[:])[:, 1]))

    num = tracer.mode_type.count(type)

    # Plot
    fig, axs = plt.subplots(int(np.ceil(num/2)), 2, figsize=(20, int(np.ceil(num/2))*3), dpi=300)
    x = np.array([-572.5, -420, -300, -180, -100, 0, 100, 260, 420,
                          572.5])  # Sensor x-coordinates - [TOWER, A03, A04, A05, A06, A07, A08, A09, A10, TOWER]
    B = 18.6  # Width of bridge girder

    phi_y_ref = ref_phi[16:32, :]
    phi_z_ref_temp = ref_phi[32:48, :]

    phi_y_ref = (modal.maxreal((phi_y_ref[::2, :] + phi_y_ref[1::2, :]) / 2))
    phi_z_ref = (modal.maxreal((phi_z_ref_temp[::2, :] + phi_z_ref_temp[1::2, :]) / 2))
    phi_t_ref = (modal.maxreal((- phi_z_ref_temp[::2, :] + phi_z_ref_temp[1::2, :]) / B))

    # Add plot of reference modes here
    f = FEM_loader.f
    phi_y_FEM = FEM_loader.phi_y
    phi_y_FEM[:, [6, 7, 14, 17]] = phi_y_FEM[:, [6, 7, 14, 17]]*(-1)
    phi_z_FEM = FEM_loader.phi_z
    phi_z_FEM[:, [1, 5, 11, 19, 20]] = phi_z_FEM[:, [1, 5, 11, 19, 20]]*(-1)
    phi_t_FEM = FEM_loader.phi_t*(-1)
    phi_t_FEM[:, [16, 21]] = phi_t_FEM[:, [16, 21]]*(-1)
    x_FEM = FEM_loader.x_plot

    j = 0
    for i in range(len(f)):
        axs[int(np.floor(j / 2)), j % 2].set_xlabel('x[m]')
        axs[int(np.floor(j / 2)), j % 2].set_ylim([-1, 1])
        axs[int(np.floor(j / 2)), j % 2].set_xlim([-600, 600])
        axs[int(np.floor(j / 2)), j % 2].set_xticks([-600, -300, 0, 300, 600])
        axs[int(np.floor(j / 2)), j % 2].set_yticks([-1, -0.5, 0, 0.5, 1])

        if FEM_loader.mode_type[i] == 'Horizontal' and type == 'Horizontal':
            factor = 1 / np.max(np.abs(phi_y_FEM[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x_FEM, phi_y_FEM[:, i] * factor, color='black', alpha=0.5)
            j += 1
        elif FEM_loader.mode_type[i] == 'Vertical' and type == 'Vertical':
            factor = 1 / np.max(np.abs(phi_z_FEM[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x_FEM, phi_z_FEM[:, i] * factor, color='black', alpha=0.5)
            j += 1
        elif FEM_loader.mode_type[i] == 'Torsional' and type == 'Torsional':
            factor = 1 / np.max(np.abs(phi_t_FEM[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x_FEM, phi_t_FEM[:, i] * factor, color='black', alpha=0.5)
            j += 1
        elif FEM_loader.mode_type[i] == 'Cable' and type == 'Cable':
            factor = 1 / np.max(np.abs(phi_y_FEM[:, i]))
            axs[int(np.floor(j / 2)), j % 2].plot(x_FEM, phi_y_FEM[:, i] * factor, color='black', alpha=0.5)
            j += 1

    for a in range(all_modeshapes.shape[1]):

        phi = all_modeshapes[:, a, :].transpose()

        phi_x, phi_y, phi_z_temp = np.split(phi, 3, axis=0)
        phi_y = phi_y[:16, :]
        phi_z_temp = phi_z_temp[:16, :]

        phi_y = (modal.maxreal((phi_y[::2, :] + phi_y[1::2, :]) / 2))
        phi_z = (modal.maxreal((phi_z_temp[::2, :] + phi_z_temp[1::2, :]) / 2))
        phi_t = (modal.maxreal((-phi_z_temp[::2, :] + phi_z_temp[1::2, :]) / B))

        j = 0
        for i in range(len(tracer.reference_modes)):
            axs[int(np.floor(j / 2)), j % 2].set_xlabel('x[m]')
            axs[int(np.floor(j / 2)), j % 2].set_ylim([-1, 1])
            axs[int(np.floor(j / 2)), j % 2].set_xlim([-600, 600])
            axs[int(np.floor(j / 2)), j % 2].set_xticks([-600, -300, 0, 300, 600])
            axs[int(np.floor(j / 2)), j % 2].set_yticks([-1, -0.5, 0, 0.5, 1])


            if tracer.mode_type[i] == 'Horizontal' and type == 'Horizontal':
                factor = 1 / np.max(np.abs(phi_y[:, i]))
                factor_ref = 1 / np.max(np.abs(phi_y_ref[:, i]))

                if np.sum(np.abs(phi_y[:, i] - phi_y_ref[:, i]*factor_ref)) > 5.0:
                    phi_y[:, i] = phi_y[:, i]*(-1)

                axs[int(np.floor(j/2)), j % 2].plot(x, np.concatenate((np.array([0]), phi_y[:, i], np.array([0])))
                                                    *factor, color='tab:red', alpha = 0.05)
                axs[int(np.floor(j / 2)), j % 2].set_title(
                    'Mode ' + str(i + 1) + ' - ' + type + '\n $\overline{f}_n$ = ' + f"{f_mean[i]:.3f}" + ' Hz, '
                                                                '$\overline{\\xi}_n$ = ' + f"{xi_mean[i]:.1f}" +'%')
                axs[int(np.floor(j / 2)), j % 2].grid()
                j += 1
            elif tracer.mode_type[i] == 'Vertical' and type == 'Vertical':
                factor = 1 / np.max(np.abs(phi_z[:, i]))
                factor_ref = 1 / np.max(np.abs(phi_z_ref[:, i]))

                if np.sum(np.abs(phi_z[:, i] - phi_z_ref[:, i]*factor_ref)) > 5.0:
                    phi_z[:, i] = phi_z[:, i]*(-1)

                axs[int(np.floor(j/2)), j % 2].plot(x, np.concatenate((np.array([0]), phi_z[:, i], np.array([0])))
                                                    *factor, color='tab:blue', alpha = 0.05)
                axs[int(np.floor(j / 2)), j % 2].set_title(
                    'Mode ' + str(i + 1) + ' - ' + type + '\n $\overline{f}_n$ = ' + f"{f_mean[i]:.3f}" + ' Hz,'
                                                                ' $\overline{\\xi}_n$ = ' + f"{xi_mean[i]:.1f}" + '%')
                axs[int(np.floor(j / 2)), j % 2].grid()
                j += 1
            elif tracer.mode_type[i] == 'Torsional' and type == 'Torsional':
                factor = 1 / np.max(np.abs(phi_t[:, i]))
                factor_ref = 1 / np.max(np.abs(phi_t_ref[:, i]))

                if np.sum(np.abs(phi_t[:, i] - phi_t_ref[:, i]*factor_ref)) > 4.0:
                    phi_t[:, i] = phi_t[:, i]*(-1)

                axs[int(np.floor(j/2)), j % 2].plot(x, np.concatenate((np.array([0]), phi_t[:, i], np.array([0])))
                                                    *factor, color='tab:orange', alpha = 0.05)
                axs[int(np.floor(j / 2)), j % 2].set_title(
                    'Mode ' + str(i + 1) + ' - ' + type + '\n $\overline{f}_n$ = ' + f"{f_mean[i]:.3f}" + ' Hz,'
                                                                ' $\overline{\\xi}_n$ = ' + f"{xi_mean[i]:.1f}" + '%')
                axs[int(np.floor(j / 2)), j % 2].grid()
                j += 1
            elif tracer.mode_type[i] == 'Cable' and type == 'Cable':
                factor = 1 / np.max(np.abs(phi_y[:, i]))
                factor_ref = 1 / np.max(np.abs(phi_y_ref[:, i]))

                if np.sum(np.abs(phi_y[:, i] - phi_y_ref[:, i]*factor_ref)) > 5.0:
                    phi_y[:, i] = phi_y[:, i]*(-1)

                axs[int(np.floor(j/2)), j % 2].plot(x, np.concatenate((np.array([0]), phi_y[:, i], np.array([0])))
                                                    *factor, color='tab:green', alpha = 0.05)
                axs[int(np.floor(j / 2)), j % 2].set_title(
                    'Mode ' + str(i + 1) + ' - ' + type + '\n $\overline{f}_n$ = ' + f"{f_mean[i]:.3f}" + ' Hz, '
                                                                '$\overline{\\xi}_n$ = ' + f"{xi_mean[i]:.1f}" +'%')
                axs[int(np.floor(j / 2)), j % 2].grid()
                j += 1

    if num % 2:
        fig.delaxes(axs[int(np.ceil(num/2))-1, 1])

    fig.tight_layout()

    return fig