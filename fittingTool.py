import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tauc_lorentz_model as tlm
import intialGuessStartingPoint as gs


class Fitter:
    def __init__(self, filedata,energy=None,er_vals=None):
        # need functionality so we can fit datafiles (w,n,k) or python varrribalies (eV,er)
        if filedata is None:
            self.energy = energy
            self.er_vals = er_vals
        else:
            self.data = filedata
            #load data and create noisy w,n,k
            self.wavelengths = self.data[:, 0]
            self.n_vals = self.data[:, 1]
            self.k_vals = self.data[:, 2]
            mask = (self.wavelengths >= 0.4) & (self.wavelengths <= 0.9)
            self.wavelengths = self.wavelengths[mask]
            self.n_vals = self.n_vals[mask]
            self.k_vals = self.k_vals[mask]

            noise = 0.005
            self.n_vals = self.n_vals + np.random.normal(0, noise, self.n_vals.size)
            self.k_vals = self.k_vals + np.random.normal(0, noise, self.k_vals.size)

            #convert to energy and create er
            self.energy, self.er_vals = self.make_variables(self.wavelengths, self.n_vals, self.k_vals)

        #fitting variables
        self.popt = None
        self.tl_model = tlm.TaucLorentz()
        self.residuals = np.inf

    @staticmethod
    def make_variables(w, n, k):
        #change input um to eV and create er
        h = 6.626e-34
        c = 299792458
        w_m = w * 1e-6
        J = (h * c) / w_m
        eV = J * 6.241509e18

        N = n + 1j * k
        er = N ** 2
        return eV, er

    #helper function to help with real vs imag
    def combined_model(self, omega, *params):
        er_model = self.tl_model.Lorentz_oscillator_model(omega, *params)
        return np.concatenate((er_model.real, er_model.imag))

    # f-sum Rule comes from a proof on Ohm’s law and continuity equation
    # adding a regulization term as the constraint
    def constraint_fj(self, params, weight=100.0):
        osc_params = params[2:]
        sum_f = 0.0
        for i in range(0, len(osc_params), 4):
            sum_f += osc_params[i + 1]
        return weight * (sum_f - 1.0)
    def constrained_model(self, omega, *params):
        # gets the models residuals
        er_model = self.tl_model.Lorentz_oscillator_model(omega, *params)
        model_vec = np.concatenate((er_model.real, er_model.imag))
        # gets the constraint residual
        constriant_residual = np.array([self.constraint_fj(params)])
        # combines it
        return np.concatenate((model_vec, constriant_residual))

    def residuals_function(self,popt):
        return np.abs(
            self.er_vals - self.tl_model.Lorentz_oscillator_model(self.energy, *popt)
        )

    def fit(self, initial_guess=None, max_iter=10, tolerance=1e-60):
        """
        Fit the Tauc-Lorentz model to the data
        to understand how we pump effects the system, we will also implement weight residual based looping for more robust fitting
        imporant note: this is not cs perfect and just grabs the popt after loop since the data even with noise it easy to fit
        """


        if initial_guess is None:
            """initial_guess = [
                1.339721, 17.299,
                2.959018719265256, 0.25, 0.12657126674208952, 2,
                2.707049876358389, 0.25, 0.37821926762804203 / 2, 2,
                2.4121183723193427, 0.25, 0.05192980617720089, 2,
                1.977398474277739, 0.75, 0.05062954042743528, 2,
            ]"""
            initial_guess = [7.12366018, 2.42366003e+00,
                             2.83093668e+00, 3.48134350e-01, 4.10094078e-01, 2.32669022e+00,
                             1.68042717e+00, 2.48187043e-03,1.42464305e-01, 5.04954721e-24,
                             2.06899415e+00, 1.28617916e-01, 4.46532433e-01, 1.55309160e+00,
                             2.35574390e+00, 5.20830783e-01,3.56175449e-01, 2.11360045e+00]

        # Initial weights
        N = len(self.energy)
        real_weights = np.ones(N)
        imag_weights = np.ones(N)
        quarter = N // 4
        real_weights[:quarter] *= 3
        real_weights[-quarter:] *= 3
        imag_weights[:quarter] *= 4
        imag_weights[-quarter:] *= 3
        weights = np.concatenate((real_weights, imag_weights))

        # Bounds
        lower_bounds = [0, 0]
        #upper_bounds = [200, np.inf] #finding ws2, either works after the fitting tool finds the fit
        upper_bounds = [2000, np.inf] # finding wse2



        osc_params = initial_guess[2:]
        length = len(osc_params)//4
        for _ in range(length):
            lower_bounds += [1, 0, 0, 0]
            upper_bounds += [5, 1, 10, 10.0]

        prev_popt = np.array(initial_guess)

        ydata_combined = np.concatenate((self.er_vals.real, self.er_vals.imag,[0]))

        best_popt = prev_popt.copy()
        best_residual_norm = np.inf
        for i in range(max_iter):
            # noinspection PyTupleAssignmentBalance
            popt, _ = curve_fit(
                self.constrained_model,
                self.energy,
                ydata_combined,
                p0=initial_guess,
                #add extra term fr residuals
                sigma=np.concatenate((1 / weights, [1])),
                bounds=(lower_bounds, upper_bounds),
                maxfev=50000
            )

            # Check change in fit parameters
            delta = np.linalg.norm(popt - prev_popt)
            print(f"Iteration {i + 1}: delta = {delta:.3e}")

            if delta < tolerance:
                print("Converged.")
                break

                # Check if this is the best fit
            if np.linalg.norm(self.residuals) < best_residual_norm:
                best_residual_norm = np.linalg.norm(self.residuals)
                best_popt = popt.copy()

            # Update weights
            self.residuals = self.residuals_function(popt)
            weights_dynamic = 1 / (1 + self.residuals + 1e-12)
            weights = np.concatenate((weights_dynamic, weights_dynamic * 5))
            prev_popt = popt

        self.popt = best_popt
        return best_popt

    def plot_fit(self):
        fitted_vals = self.tl_model.Lorentz_oscillator_model(self.energy, *self.popt)

        plt.figure(figsize=(12, 6))
        # Real part
        plt.subplot(1, 2, 1)
        plt.plot(self.energy, self.er_vals.real, 'b.', label='Measured Re(ε)')
        plt.plot(self.energy, fitted_vals.real, 'r-', label='Fitted Re(ε)')
        plt.title('Real Part of Permittivity')
        plt.xlabel('eV')
        plt.ylabel('ε_real')
        plt.legend()
        # Imaginary part
        plt.subplot(1, 2, 2)
        plt.plot(self.energy, self.er_vals.imag, 'g.', label='Measured Im(ε)')
        plt.plot(self.energy, fitted_vals.imag, 'r-', label='Fitted Im(ε)')
        plt.title('Imaginary Part of Permittivity')
        plt.xlabel('eV')
        plt.ylabel('ε_imag')
        plt.legend()
        plt.tight_layout()
        plt.show()

