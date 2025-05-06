"""
VaporFit: Atmospheric Spectrum Correction Tool
Version: 0.1
Author: Przemysław Pastwa, Piotr Bruździak
Affiliation: Gdańsk University of Technology, Department of Physical Chemistry
Contact: piotr.bruzdziak@pg.edu.pl
Date: 2025-02-13
License: GNU GPL v.3.0 + Citation Requirement

Description:
VaporFit is an open-source software tool designed for automated atmospheric
correction in FTIR spectroscopy. It employs a least squares fitting approach
using multiple reference atmospheric/vapor spectra, dynamically adjusting
subtraction coefficients to improve accuracy.

Required Libraries:
- numpy
- scipy
- matplotlib

User Guide:
For detailed instructions on using VaporFit, refer to the **User Guide**.
"""

############################################# Imports ##################################################################
import os
import tkinter as tk
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter as savitzky_golay
from scipy.optimize import least_squares as leastsq
from scipy.linalg import svd



#############################################  Initialize global variables #############################################
file_paths, atm_file_paths, wavenb, spectrum_data, atmosphere_data, residue_data = None, None, None, None, None, None
wavenb_filtered, spectra_filtered, atm_spectra_filtered, corr_spectra = None, None, None, None
fit_params_for_plot = None
atm_load, spectra_load, calculated_params = False, False, False
delzc_m, delzc_c, delzc_tau = 3, 6, 9
p_components_raw, p_components_cor, explained_variance_raw, explained_variance_cor = None, None, None, None


############################################# Classes ##################################################################
class AtmFitParams:
    """This class handles the parameters and methods for fitting atmospheric spectra to a given spectrum.
    It includes methods for calculating residuals, fitting the parameters,
    and subtracting the atmospheric spectra from the given spectrum."""
    def __init__(self,wavenb,spectrum,atm_spectra,sg_poly=3,sg_points=11):
        self.wavenb = np.asarray(wavenb,dtype=float) # list of wavenumbers
        self.spectrum = np.asarray(spectrum,dtype=float) # spectrum from which atmosphere (atm) spectra will be subtracted
        self.atm_spectra = np.asarray(atm_spectra,dtype=float) # array of atmosphere spectra
        self.no_of_atm_spctr = self.atm_spectra.shape[1] #number of vapor spectra
        self.init_fit_params = np.asarray((self.no_of_atm_spctr*[0.1]), dtype=float) # initial parameters of atm fitting
        self.no_of_parameters = np.size(self.init_fit_params) # number of parameters
        self.bounds = (self.no_of_parameters*[-np.inf],self.no_of_parameters*[np.inf]) #initial lower and upper bounds for all parameter set to +/- inf
        self.sg_poly,self.sg_points = sg_poly,sg_points # Savitzky-Golay smoothing factors

    def residuals(self,params):
        atm_sum = np.sum((params*self.atm_spectra),axis=1)
        total_sum = self.spectrum - atm_sum
        smoothed_total_sum = savitzky_golay(total_sum,self.sg_points,self.sg_poly)
        return smoothed_total_sum - total_sum  #residual minimized during fitting procedure

    def fit(self):
        fit_result = leastsq(self.residuals,self.init_fit_params,ftol=1.49012e-10, xtol=1.49012e-10,bounds=self.bounds)
        return fit_result.x # Fitted parameters of water vapor subtraction (baseline - last 3)

    def atm_subtract(self):
        atm_sum = np.sum((self.fit()*self.atm_spectra),axis=1)
        return self.spectrum - atm_sum # Vapor-corrected spectrum


class ToolTip:
    """This class creates and manages tooltips for Tkinter widgets.
    It handles the display and hiding of tooltips with a specified delay."""
    def __init__(self, widget, text, delay=2000):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.delay = delay
        self.after_id = None

        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def schedule_tooltip(self, event=None):
        """Scheduling the display of a tooltip after a specified time."""
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self):
        """Creates and displays tooltips."""
        if self.tooltip_window or not self.widget.winfo_ismapped():
            return

        x, y, _, _ = self.widget.bbox("insert")  # initial position
        x += self.widget.winfo_rootx() + 20  # Cursor offset
        y += self.widget.winfo_rooty() + 20

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window border
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify='left', background="lightyellow", relief="solid", borderwidth=1)
        label.pack(ipadx=5, ipady=2)

    def hide_tooltip(self, event=None):
        """Hides tooltips."""
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

############################################# File operations ##########################################################
def load_files(atm=False):
    """This function loads spectrum files (CSV, DPT) and returns the wavenumbers and spectra.
    It also updates the global variables related to loaded spectra and atmospheric data."""
    global atm_load, spectra_load, atmosphere_data, spectrum_data
    file_paths = tk.filedialog.askopenfilenames(filetypes=[("Data files", "*.csv *.dpt")])
    if file_paths:
        try:
            data_arrays = [np.loadtxt(file, delimiter=",") for file in file_paths]  # Load CSV files as NumPy arrays
            wavenb = data_arrays[0][:, 0]  # First column from the first file
            spectra = np.column_stack([data[:, 1] for data in data_arrays])  # Second column from each file, stacked
            if not atm:
                if atm_load:
                    plot_spectral_data(wavenb, None , spectra, atmosphere_data)
                else:
                    plot_spectral_data(wavenb, None, spectra, None)
                spectra_load = True
            else:
                if spectra_load:
                    plot_spectral_data(wavenb, None, spectrum_data, spectra)
                else:
                    plot_spectral_data(wavenb, None, None, spectra)
                atm_load = True
            return wavenb, spectra, file_paths
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error loading files: {str(e)}")
    return None, None, None


def load_spectra_files():
    """This function allows the user to select spectrum files and loads them using the load_spectrum_files function.
    It also clears previously loaded atmospheric spectra from memory."""
    global wavenb, spectrum_data, file_paths, atmosphere_data, atm_file_paths, atm_load
    # Clear previously loaded atmospheric spectra
    atmosphere_data = None
    atm_file_paths = None
    atm_load = False

    # Load new spectrum files
    wavenb, spectrum_data, file_paths = load_files(atm=False)
    # if spectrum_data is not None:
    #    tk.messagebox.showinfo("Info", "Spectrum files loaded successfully.")


def load_atmospheric_spectra_files():
    """This function allows the user to select atmospheric files and loads them using the load_spectrum_files function."""
    global atmosphere_data, atm_file_paths
    _, atmosphere_data, atm_file_paths = load_files(atm=True)
    #if atmosphere_data is not None:
    #    tk.messagebox.showinfo("Info", "Atmosphere files loaded successfully.")


############################################# Spectra processing  and correction #######################################
def filter_wavenumber_range(wavenb, spectra, min_wavenb, max_wavenb):
    """This function filters the wavenumbers and spectra based on the specified minimum and maximum wavenumber values."""
    mask = (wavenb >= min_wavenb) & (wavenb <= max_wavenb)
    return np.asarray(wavenb[mask]), np.asarray(spectra[mask, :])


def perform_correction():
    """This function performs the vapor correction on the loaded spectra using the specified Savitzky-Golay parameters.
    It updates the corrected spectra, exports the corrected files, and updates the parameters display."""
    global wavenb, fit_params_for_plot, wavenb_filtered, spectra_filtered, atm_spectra_filtered, corr_spectra
    try:
        plt.close() #close previous plots
        list_of_spectra_objects, corr_spectra = [], [] #create placeholders for spectra objects and corrected spectra
        sg_poly, sg_points = int(entry_widgets["SG polynomial:"].get()), int(entry_widgets["SG points:"].get())  # get S-G parameters

        try:
            min_wavenb = float(entry_widgets["Min wavenumber:"].get())
            max_wavenb = float(entry_widgets["Max wavenumber:"].get())
        except ValueError:
            min_wavenb, max_wavenb = wavenb[0], wavenb[-1]

        wavenb_filtered, spectra_filtered = filter_wavenumber_range(wavenb, spectrum_data, min_wavenb, max_wavenb)
        _,atm_spectra_filtered = filter_wavenumber_range(wavenb, atmosphere_data, min_wavenb, max_wavenb)

        no_of_spectra = spectra_filtered.shape[1]
        for i in range(no_of_spectra):
            list_of_spectra_objects.append(
                AtmFitParams(wavenb_filtered, spectra_filtered[:, i], atm_spectra_filtered, sg_poly=sg_poly, sg_points=sg_points))
            corr_spectra.append(list_of_spectra_objects[i].atm_subtract())
        corr_spectra = np.asarray(corr_spectra).T

        fit_params_for_plot = []
        for i,spec in enumerate(list_of_spectra_objects):
             fit_params_for_plot.append(spec.fit())
        fit_params_for_plot = np.array(fit_params_for_plot)


        # Update plot
        plot_spectral_data(wavenb_filtered, corr_spectra, spectra_filtered, atm_spectra_filtered)

    except Exception as e:
        tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")


############################################# Data analysis ############################################################
def calculate_spectral_smoothness(spectrum):
    """This function calculates the Spectral Smoothness Index (SSI) for a given spectrum.
    A lower value indicates greater smoothness."""
    diff_sq_sum = np.sum(np.diff(spectrum)**2)
    energy = np.sum(spectrum**2)
    return diff_sq_sum / energy if energy > 0 else 0


def calculate_smoothness_variance(wavenumbers, spectrum):
    """This function calculates the variance of the second derivative (curvature smoothness) for a given spectrum.
    A lower value indicates greater smoothness."""
    second_derivative = np.gradient(np.gradient(spectrum, wavenumbers), wavenumbers)
    return np.var(second_derivative)


def calculate_std_after_smoothing(spectrum):
    """This function calculates the standard deviation of the residue after smoothing a given spectrum
    using the Savitzky-Golay filter."""
    smoothed_spectrum = savitzky_golay(spectrum, window_length=11, polyorder=3)
    residue = spectrum - smoothed_spectrum
    return np.std(residue)


def normalize(y):
    """This function normalizes a given array to the range [0, 1]."""
    return (y - np.min(y)) / (np.max(y) - np.min(y))


def find_optimal_sg_parameters():
    """This function performs an optimum search for the Savitzky-Golay window size
    by evaluating different parameters and plotting the results."""
    global wavenb, spectrum_data, atmosphere_data

    sg_poly, sg_points = int(entry_widgets["SG polynomial:"].get()), int(entry_widgets["SG points:"].get())
    # Create a list of xx values around sg_points, with intervals of 2
    param_list = [sg_points + i * 2 for i in range(-4, 5)]
    # Remove values smaller than (sg_poly + 2)
    min_valid = sg_poly + 2
    param_list = [val for val in param_list if val >= min_valid]
    # Fill in missing values from the top
    while len(param_list) < 11:
        param_list.append(param_list[-1] + 2)

    try:
        min_wavenb = float(entry_widgets["Min wavenumber:"].get())
        max_wavenb = float(entry_widgets["Max wavenumber:"].get())
    except ValueError:
        min_wavenb, max_wavenb = wavenb[0], wavenb[-1]

    ssi_sums = []
    cv_sums = []
    #delzc_sums = []
    std_sums = []

    for param in param_list:
        try:
            list_of_spectra_objects, spectra_corrected = [], []  # create placeholders for spectra objects and corrected spectra
            wavenb_filtered, spectra_filtered = filter_wavenumber_range(wavenb, spectrum_data, min_wavenb, max_wavenb)
            _, atm_spectra_filtered = filter_wavenumber_range(wavenb, atmosphere_data, min_wavenb, max_wavenb)

            no_of_spectra = spectra_filtered.shape[1]
            for i in range(no_of_spectra):
                list_of_spectra_objects.append(
                    AtmFitParams(wavenb_filtered, spectra_filtered[:, i], atm_spectra_filtered, sg_poly=sg_poly, sg_points=param))
                spectra_corrected.append(list_of_spectra_objects[i].atm_subtract())
            spectra_corrected = np.asarray(spectra_corrected).T

            ssi_temp,cv_temp,delzc_temp,std_temp = [],[],[],[]
            for i in range(no_of_spectra):      #smoothness parameters
                cv_temp.append(calculate_smoothness_variance(wavenb_filtered, spectra_corrected[:,i]))
                ssi_temp.append(calculate_spectral_smoothness(spectra_corrected[:,i]))
                #delzc_temp.append(calculate_delzc(spectra_corrected[:,i]))
                std_temp.append(calculate_std_after_smoothing(spectra_corrected[:,i]))

            ssi_sums.append(np.sum(ssi_temp))
            cv_sums.append(np.sum(cv_temp))
            #delzc_sums.append(np.sum(delzc_temp))
            std_sums.append(np.sum(std_temp))

        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Finding the indices of the minimum.
    min_ssi_index = ssi_sums.index(min(ssi_sums))
    min_cv_index = cv_sums.index(min(cv_sums))
    #min_delzc_index = delzc_sums.index(min(delzc_sums))
    min_std_index = std_sums.index(min(std_sums))

    # Finding the corresponding values in param_list
    min_ssi_param = param_list[min_ssi_index]
    min_cv_param = param_list[min_cv_index]
    #min_delzc_param = param_list[min_delzc_index]
    min_std_param = param_list[min_std_index]

    plot_window = tk.Toplevel(root)
    plot_window.title("Optimum polynomial search")
    plot_window.geometry("700x500")
    # TCreate graph
    fig, ax1 = plt.subplots(dpi=80)
    #ax2 = ax1.twinx()

    ax1.plot(param_list, normalize(std_sums), 'r-', linewidth=2, label='SD min = {}'.format(int(min_std_param)))
    ax1.plot(param_list, normalize(cv_sums), 'b--', linewidth=0.5, label='SDV min = {}'.format(int(min_cv_param)))
    ax1.plot(param_list, normalize(ssi_sums), 'g--', linewidth=0.5, label='SSI min = {}'.format(int(min_ssi_param)))
    #ax1.plot(param_list, normalize(delzc_sums), 'k--', linewidth=0.5, label='DELZC min = {}'.format(int(min_delzc_param)))

    ax1.legend(loc='best')
    ax1.set_xlabel('Window width')
    ax1.set_title('Optimum SG points search for SG poly={}'.format(sg_poly))

    fig.tight_layout()
    #plt.show()

    # Place graphs in tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def perform_pca_analysis():
    """This function performs Principal Component Analysis (PCA) on the filtered and corrected spectra.
    It plots the explained variance and principal components before and after correction."""
    global wavenb_filtered, spectra_filtered, atm_spectra_filtered, corr_spectra, p_components_raw, p_components_cor, explained_variance_raw, explained_variance_cor, entry_widgets, root
    pcs = int(entry_widgets["PC count:"].get())  # number of principal components
    tk.messagebox.showinfo("Alert!", "Make sure to run correction first!")

    if corr_spectra is not None:
        no_of_spectra = spectra_filtered.shape[1]

        # Standaryzacja danych
        spectra_mean = np.mean(spectra_filtered, axis=1, keepdims=True)
        corr_spectra_mean = np.mean(corr_spectra, axis=1, keepdims=True)
        spectra_centered = spectra_filtered - spectra_mean
        corr_spectra_centered = corr_spectra - corr_spectra_mean

        # SVD
        U1, S1, Vt1 = svd(spectra_centered.T, full_matrices=False)
        U2, S2, Vt2 = svd(corr_spectra_centered.T, full_matrices=False)

        # Explained variance
        explained_variance_raw = S1 ** 2 / (no_of_spectra - 1)
        explained_variance_cor = S2 ** 2 / (no_of_spectra - 1)

        # select number of PCs
        p_components_raw = Vt1[:pcs].T
        p_components_cor = Vt2[:pcs].T

        plot_window = tk.Toplevel(root)
        plot_window.title("PCA")
        plot_window.geometry("1000x500")

        x_values = np.arange(1, no_of_spectra + 1, 1)

        # Create graph window
        fig, ax = plt.subplots(2, 2, dpi=80, gridspec_kw={'height_ratios': [1, 2]})
        ax[0, 0].set_xlabel('PCA number')
        ax[0, 0].set_ylabel('Explained variance')
        ax[0, 0].set_title('Before correction')
        ax[0, 0].plot(x_values[:-2], explained_variance_raw[:-2], 'o-', color='r', label='before')
        ax[0, 0].legend(loc='best')
        plt.tight_layout()

        ax[0, 1].set_xlabel('PCA number')
        ax[0, 1].set_ylabel('Explained variance')
        ax[0, 1].set_title('After correction')
        ax[0, 1].plot(x_values[:-2], explained_variance_cor[:-2], 'o-', color='r', label='after')
        ax[0, 1].legend(loc='best')
        plt.tight_layout()

        initial_xlim = (max(wavenb_filtered), min(wavenb_filtered))
        initial_ylim = {}


        zoom_rects = {}  # Selection rectangles for each plot
        zoom_start = {}  # Initial click coordinates

        ax[1, 0].set_xlabel('Wavenumber ($cm^{-1}$)')
        ax[1, 0].plot(wavenb_filtered, p_components_raw, label='before, {} PC'.format(pcs))
        ax[1, 0].set_xlim(*initial_xlim)
        ax[1, 0].legend(loc='best')
        plt.tight_layout()

        ax[1, 1].set_xlabel('Wavenumber ($cm^{-1}$)')
        ax[1, 1].plot(wavenb_filtered, p_components_cor, label='after, {} PC'.format(pcs))
        ax[1, 1].set_xlim(*initial_xlim)
        ax[1, 1].legend(loc='best')
        plt.tight_layout()

        ax_bottom_left = ax[1, 0]
        ax_bottom_right = ax[1, 1]
        bottom_axes = [ax_bottom_left, ax_bottom_right]

        for ax_b in bottom_axes: # use ax_b to avoid shadowing outer ax variable
            initial_ylim[ax_b] = ax_b.get_ylim()

        zoom_rects = {}  # Select rectangles for each plot
        zoom_start = {}  # Initial click coordinates

        def on_button_press(event):
            """Starting to draw a rectangle by holding down the left mouse button, only on the bottom plots."""
            ax = event.inaxes
            if ax not in bottom_axes or event.button != 1: # check if the click is in the bottom axes
                return

            zoom_start[ax] = (event.xdata, event.ydata)

            if ax not in zoom_rects:
                zoom_rects[ax] = Rectangle((event.xdata, event.ydata), 0, 0, linewidth=1, edgecolor='black',
                                           linestyle='dashed', facecolor='none')
                ax.add_patch(zoom_rects[ax])

        def on_motion(event):
            """Drawing a rectangle while dragging, only on the bottom plots."""
            ax = event.inaxes
            if ax not in bottom_axes or ax not in zoom_start or event.xdata is None or event.ydata is None: # check if motion is in bottom axes and zoom started there
                return

            x0, y0 = zoom_start[ax]
            width = event.xdata - x0
            height = event.ydata - y0

            zoom_rects[ax].set_width(width)
            zoom_rects[ax].set_height(height)
            zoom_rects[ax].set_xy((x0, y0))
            canvas.draw()

        def on_button_release(event):
            """Zooming into the selected area on both bottom plots, only if the action started on a bottom plot."""
            ax = event.inaxes
            if ax not in bottom_axes or ax not in zoom_rects or event.button != 1: # check if release is in bottom axes and zoom started there
                return

            x0, y0 = zoom_start[ax]
            x1, y1 = event.xdata, event.ydata

            if x0 is None or y0 is None or x1 is None or y1 is None:
                return

            if abs(x1 - x0) > 1e-6 and abs(y1 - y0) > 1e-6:
                new_xlim = (max(x0, x1), min(x0, x1))
                new_ylim = (min(y0, y1), max(y0, y1))

                # Set new range on both bottom plots
                ax_bottom_left.set_xlim(new_xlim)
                ax_bottom_right.set_xlim(new_xlim)
                ax_bottom_left.set_ylim(new_ylim)
                ax_bottom_right.set_ylim(new_ylim)

            # Remove the selection rectangle
            zoom_rects[ax].remove()
            del zoom_rects[ax]
            del zoom_start[ax]

            canvas.draw()

        def on_right_click(event):
            """Reset both bottom plots after right-click, only if right click is on bottom axes."""
            ax = event.inaxes
            if ax not in bottom_axes or event.button != 3: # check if right click is in bottom axes
                return

            # Reset to the initial range for both plots
            ax_bottom_left.set_xlim(*initial_xlim)
            ax_bottom_right.set_xlim(*initial_xlim)
            ax_bottom_left.set_ylim(*initial_ylim[ax_bottom_left])
            ax_bottom_right.set_ylim(*initial_ylim[ax_bottom_right])

            canvas.draw()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        fig.canvas.mpl_connect('button_press_event', on_button_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_button_release)
        fig.canvas.mpl_connect('button_press_event', on_right_click) # Corrected event type to 'button_press_event' for right click

    else:
        tk.messagebox.showinfo("Error", "Correction not yet completed.")

def display_parameters_plot():
    """This function creates a plot of the subtraction parameters and displays it in a Tkinter window."""
    global fit_params_for_plot
    if fit_params_for_plot is not None:
        plot_window = tk.Toplevel(root)
        plot_window.title("Subtraction parameters")
        plot_window.geometry("700x500")

        x_values = np.arange(1, fit_params_for_plot.shape[0] + 1, 1)
        # create graphs
        fig, ax = plt.subplots(dpi=80)
        for i in range(fit_params_for_plot.shape[1]):  # columnwise iteration
            ax.scatter(x_values, fit_params_for_plot[:, i], label="a.s. {}".format(i + 1))
        ax.set_title("Subtraction parameters")
        ax.set_xlabel('Spectrum index')
        ax.set_ylabel('Subtraction parameters')
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()

        # creates graph in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        tk.messagebox.showinfo("Error", "Correction not yet completed.")


def plot_spectral_data(wavenb, corr_spectra=None, spectra_filtered=None, atmosphere=None):
    """Rysowanie widm z możliwością zaznaczania obszaru do powiększenia oraz resetowania widoku."""

    for widget in plot_frame.winfo_children():
        widget.destroy()  # remove previous graphs
    fig = plt.figure(dpi=80)
    fig.tight_layout()

    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2:])

    initial_xlim = (max(wavenb), min(wavenb))
    initial_ylim = {}

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(*initial_xlim)

    if spectra_filtered is not None:
        for i in range(spectra_filtered.shape[1]):
            ax1.plot(wavenb, spectra_filtered[:, i], linewidth=1, label=f'Spectrum {i+1}')
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    if atmosphere is not None:
        for i in range(atmosphere.shape[1]):
            ax2.plot(wavenb, atmosphere[:, i], linewidth=1, label=f'Vapor spec. {i+1}')
        ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, right=True, labelright=True, left=False, labelleft=False)

    if corr_spectra is not None:
        for i in range(corr_spectra.shape[1]):
            ax3.plot(wavenb, corr_spectra[:, i], linewidth=1.5, label=f'Spectrum {i+1}')
            ax3.plot(wavenb, spectra_filtered[:, i], 'k--', linewidth=0.5)
        ax3.set_xlabel('Wavenumber ($cm^{-1}$)')

    for ax in [ax1, ax2, ax3]:
        initial_ylim[ax] = ax.get_ylim()

    plt.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07, hspace=0.1, wspace=0.1)

    zoom_rects = {}  # Selection rectangles for each plot
    zoom_start = {}  # Initial click coordinates

    def on_button_press(event):
        """Start drawing rectangle while holding the left mouse button."""
        ax = event.inaxes
        if ax is None or event.button != 1:
            return

        zoom_start[ax] = (event.xdata, event.ydata)

        if ax not in zoom_rects:
            zoom_rects[ax] = Rectangle((event.xdata, event.ydata), 0, 0, linewidth=1, edgecolor='black', linestyle='dashed', facecolor='none')
            ax.add_patch(zoom_rects[ax])

    def on_motion(event):
        """Draw the rectangle while dragging."""
        ax = event.inaxes
        if ax is None or ax not in zoom_start or event.xdata is None or event.ydata is None:
            return

        x0, y0 = zoom_start[ax]
        width = event.xdata - x0
        height = event.ydata - y0

        zoom_rects[ax].set_width(width)
        zoom_rects[ax].set_height(height)
        zoom_rects[ax].set_xy((x0, y0))
        canvas.draw()

    def on_button_release(event):
        """Zoom into the selected area after releasing the left mouse button."""
        ax = event.inaxes
        if ax is None or ax not in zoom_rects or event.button != 1:
            return

        x0, y0 = zoom_start[ax]
        x1, y1 = event.xdata, event.ydata

        if x0 is None or y0 is None or x1 is None or y1 is None:
            return

        # Check if the selected area has a reasonable size
        if abs(x1 - x0) > 1e-6 and abs(y1 - y0) > 1e-6:
            ax.set_xlim(max(x0, x1), min(x0, x1))
            ax.set_ylim(min(y0, y1), max(y0, y1))

        # Remove the rectangle
        zoom_rects[ax].remove()
        del zoom_rects[ax]
        del zoom_start[ax]

        canvas.draw()

    def on_right_click(event):
        """Reset the plot after right-clicking."""
        ax = event.inaxes
        if ax is None or event.button != 3:
            return

        ax.set_xlim(*initial_xlim)
        ax.set_ylim(*initial_ylim[ax])
        canvas.draw()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    fig.canvas.mpl_connect('button_press_event', on_button_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_button_release)
    fig.canvas.mpl_connect('button_press_event', on_right_click)


def save_analysis_report():
    global file_paths, atm_file_paths, wavenb_filtered, spectra_filtered, atm_spectra_filtered, corr_spectra, \
        fit_params_for_plot, explained_variance_raw, explained_variance_cor, p_components_raw, p_components_cor

    file_path = tk.filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")],
        title="Save CSV Report Files"
    )

    if not file_path:
        return

    base_path = file_path.rsplit('.', 1)[0]

    # "Information" sheet
    info_data = [
        ["Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Spectra count", spectra_filtered.shape[1] if spectra_filtered is not None else 0],
        ["ATM spectra count", atm_spectra_filtered.shape[1] if atm_spectra_filtered is not None else 0],
        ["Min wavenumber", float(entry_widgets["Min wavenumber:"].get())],
        ["Max wavenumber", float(entry_widgets["Max wavenumber:"].get())],
        ["SG Poly", int(entry_widgets["SG polynomial:"].get())],
        ["SG Points", int(entry_widgets["SG points:"].get())],
        ["", ""],  # Pusta linia
        ["Spectra Files", ""]
    ]

    if file_paths:
        info_data.extend([[f"Spectrum {i + 1}", file] for i, file in enumerate(file_paths)])

    info_data.append(["", ""])  # blank
    info_data.append(["ATM Spectra Files", ""])

    if atm_file_paths:
        info_data.extend([[f"ATM Spectrum {i + 1}", file] for i, file in enumerate(atm_file_paths)])

    np.savetxt(f"{base_path}_Information.csv", info_data, delimiter=",", fmt="%s")

    # "Corrected spectra" sheet
    if corr_spectra is not None:
        corr_data = np.column_stack((wavenb_filtered, corr_spectra))
        np.savetxt(f"{base_path}_Corrected_spectra.csv", corr_data, delimiter=",", header="Wavenumbers / cm-1," + ",".join([f"Spectrum {i + 1}" for i in range(corr_spectra.shape[1])]), comments='')

    # "Original spectra" sheet
    if spectra_filtered is not None:
        orig_data = np.column_stack((wavenb_filtered, spectra_filtered))
        np.savetxt(f"{base_path}_Original_spectra.csv", orig_data, delimiter=",", header="Wavenumbers / cm-1," + ",".join([f"Spectrum {i + 1}" for i in range(spectra_filtered.shape[1])]), comments='')

    # "ATM spectra" sheet
    if atm_spectra_filtered is not None:
        atm_data = np.column_stack((wavenb_filtered, atm_spectra_filtered))
        np.savetxt(f"{base_path}_ATM_spectra.csv", atm_data, delimiter=",", header="Wavenumbers / cm-1," + ",".join([f"ATM Spectrum {i + 1}" for i in range(atm_spectra_filtered.shape[1])]), comments='')

    # "Correction parameters" sheet
    if fit_params_for_plot is not None:
        params_header = ",".join([f"ATM {i + 1}" for i in range(fit_params_for_plot.shape[1])])
        params_index = [f"Spectrum {i + 1}" for i in range(fit_params_for_plot.shape[0])]
        params_data = np.column_stack((params_index, fit_params_for_plot))
        np.savetxt(f"{base_path}_Correction_parameters.csv", params_data, delimiter=",", header="," + params_header, comments='', fmt="%s")

    # "PCA - EV" sheet
    if explained_variance_raw is not None and explained_variance_cor is not None:
        pca_ev_data = np.column_stack((np.arange(1, len(explained_variance_raw) + 1), explained_variance_raw, explained_variance_cor))
        np.savetxt(f"{base_path}_PCA_EV.csv", pca_ev_data, delimiter=",", header="Component,EV Before Correction,EV After Correction", comments='')

    # "PCA - Components" sheet
    if p_components_raw is not None and p_components_cor is not None:
        pca_comp_data = np.column_stack((wavenb_filtered, p_components_raw, p_components_cor))
        pca_comp_header = "Wavenumbers / cm-1," + ",".join([f"PC{i + 1} Before Correction" for i in range(p_components_raw.shape[1])]) + "," + ",".join([f"PC{i + 1} After Correction" for i in range(p_components_cor.shape[1])])
        np.savetxt(f"{base_path}_PCA_Components.csv", pca_comp_data, delimiter=",", header=pca_comp_header, comments='')

    tk.messagebox.showinfo("Success", "CSV report saved successfully.")


def save_corrected_spectra_as_csv():
    global file_paths, wavenb_filtered, corr_spectra
    # export corrected files to *.csv
    try:
        for i, file_path in enumerate(file_paths):
            dir_path = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(dir_path, f"corr_{base_name}.csv")
            corrected_data = np.column_stack((wavenb_filtered, corr_spectra[:, i]))
            np.savetxt(output_path, corrected_data, delimiter=",", fmt="%.6f")
        tk.messagebox.showinfo("Success", "CSV files saved successfully\nin the same directory as the imported spectra.")
    except Exception as e:
        tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")


def display_about_info():
    """Displays information about the program."""
    tk.messagebox.showinfo("About", "VaporFit v.0.1\nThis program was designed for vapor correction of FTIR spectra.\n"
                                    "Please refer to the User Guide for more information.\n"
                                    "Contact: piotr.bruzdziak@pg.edu.pl")


def exit_application():
    """This function closes the Tkinter application."""
    root.destroy()
    root.quit()

############################################# TKINTER SETUP ############################################################
############################################# Tkinter setup main window ################################################
root = tk.Tk()
root.title("Vaporfit - Vapor Correction Tool")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry('{}x{}'.format(int(screen_width*0.8), int(screen_height*0.8)))

############################################# Create a menu bar ########################################################
menu_bar = tk.Menu(root)
about_menu = tk.Menu(menu_bar, tearoff=0)
#about_menu.add_command(label="Help", command=display_help_dialog)
about_menu.add_command(label="About this program", command=display_about_info)
menu_bar.add_cascade(label="About", menu=about_menu)
root.config(menu=menu_bar)

############################################# Create buttons ###########################################################
frame = tk.Frame(root)
frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")
buttons = [
    ("Load\nSpectra", load_spectra_files, "Load spectra for correction (*.csv, *.dtp).\nAll parameters of all uploaded spectra (wavenumber range, resolution, number of points, etc.) must be identical!"),
    ("Load\nATM Files", load_atmospheric_spectra_files, "Load atmospheric spectra for subtraction (*.csv, *.dtp).\nAll parameters of all uploaded spectra (wavenumber range, resolution, number of points, etc.) must be identical!"),
    ("Correct!\n", perform_correction, "Apply correction."),
    ("Find\nOptimum", find_optimal_sg_parameters, "Display graph of smoothness parameters\nto find optimal SG values."),
    ("Correction\nParameters", display_parameters_plot, "Show parameters for atmospheric spectra subtraction."),
    ("PCA\n", perform_pca_analysis, "Perform PCA analysis (EV) and show a selected number\nof first PCs before and after correction."),
    ("Save\nreport", save_analysis_report, "Export report to multiple CSV files: all spectra,\ncorrection parameters, and PCA results."),
    ("Save\nspectra", save_corrected_spectra_as_csv, "Export the corrected spectra in CSV format to the same folder as the original spectra."),
    ("Close\n", exit_application, "Close application.")
]
for i, (text, command, tooltip) in enumerate(buttons):
    btn = tk.Button(frame, text=text, command=command)
    btn.grid(row=i, column=0, padx=5, pady=5, sticky='ew')
    ToolTip(btn, tooltip, delay=1000)

############################################# Create inputs ###########################################################
settings_frame = tk.Frame(root)
settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="sew")
labels_entries = [
    ("SG polynomial:", "3"),
    ("SG points:", "11"),
    ("Min wavenumber:", "1000"),
    ("Max wavenumber:", "2000"),
    ("PC count:", "2")
]
entry_widgets = {}

tooltips = {
    "SG polynomial:": "Degree of the polynomial used in Savitzky-Golay filter.",
    "SG points:": "Number of points used in Savitzky-Golay filter.",
    "Min wavenumber:": "Minimum wavenumber for analysis.",
    "Max wavenumber:": "Maximum wavenumber for analysis.",
    "PC count:": "Number of principal components to retain.\nMust be lower or equal to the number of spectra!\nRun correction first!"
}

for i, (label_text, default_value) in enumerate(labels_entries):
    tk.Label(settings_frame, text=label_text).grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(settings_frame, width=5)
    entry.insert(0, default_value)
    entry.grid(row=i, column=1, padx=5, pady=5, sticky='e')
    entry_widgets[label_text] = entry
    ToolTip(entry, tooltips[label_text], delay=1000)
############################################# Create plots #############################################################
plot_frame = tk.Frame(root)
plot_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

############################################# Configure grid layout ####################################################
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)

############################################# Run Tkinter ##############################################################
root.mainloop()
