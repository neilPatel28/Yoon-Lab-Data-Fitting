import numpy as np
import torch
import matplotlib.pyplot as plt
import tauc_lorentz_model as tlm
from scipy.signal import find_peaks, peak_widths

# NOTE: this code was aided by northeastern provided claude account to allow me, to refactored
# loop based, class-focused code into a more modern, vectorized torch implementation.

class ReflectanceCalculator:
    """
    Class to calculate reflectance curve (RC).
    """

    def __init__(self, d1=5, d2=5, d5=13, d6=5, delta1=0,delta2=0,delta3=0,delta4=0,device="cpu"):
        """
        Initialize with layer thicknesses.
        """
        self.device = device

        # Auto-detect CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = "cpu"

        self.d1 = d1
        self.d2 = d2
        self.d5 = d5
        self.d6 = d6



        # Load data into tensor
        self.data = self.load_data(delta1=0, delta2=0, delta3=0, delta4=0)

    def load_data(self,delta1=0, delta2=0, delta3=0, delta4=0):
        """
        Load and prepare all optical data from files as PyTorch tensor.
        """
        # Load hBN (reference wavelength grid)
        data = np.loadtxt("hbnindex.txt")
        wavelengths = data[:, 0]  # Keep in micrometers for now
        hbnN = data[:, 1]
        hbnK = data[:, 2]

        # Load graphene (carbon)
        data = np.loadtxt("grapheneindex.txt")
        cW = data[:, 0]
        cN = data[:, 1]
        cK = data[:, 2]

        # Load WSe2
        data = np.loadtxt("wse2index.txt")
        wse2W = data[:, 0]
        wse2N = data[:, 1]
        wse2K = data[:, 2]

        # Load WS2
        data = np.loadtxt("ws2index.txt")
        ws2W = data[:, 0]
        ws2N = data[:, 1]
        ws2K = data[:, 2]

        # Load Si
        data = np.loadtxt("siindex.txt")
        siW = data[:, 0]
        siN = data[:, 1]
        siK = data[:, 2]

        # Load SiO2
        data = np.loadtxt("sio2index.txt")
        sio2W = data[:, 0]
        sio2N = data[:, 1]
        sio2K = np.zeros_like(sio2N)


        #delta shift
        # replace the raw data with the new data
        ws2energy = 0.00000123982884337 / (ws2W * 1e-6)
        ws2_exp_params = np.array([1.40297001e+02, 1.15183473e+01,
                                   2.95616242e+00, 2.92483712e-01, 4.27646826e-01, 1.99807325e+00,
                                   # change to match the experiment
                                   1.97 + delta1, 4.71331035e-01, 0.007 + delta2, 1.80029956e+00,
                                   2.37185291e+00, 1.18930408e-01, 2.28297482e-01, 1.87449362e+00,
                                   2.69456857e+00, 1.17254844e-01, 2.23229104e-01, 1.73045554e+00])
        tl_model = tlm.TaucLorentz()
        er_model_exp = tl_model.Lorentz_oscillator_model(ws2energy, *ws2_exp_params)
        e1 = er_model_exp.real
        e2 = er_model_exp.imag

        ws2N = (1 / np.sqrt(2)) * np.sqrt(e1 + np.sqrt(e1 ** 2 + e2 ** 2))
        ws2K = (1 / np.sqrt(2)) * np.sqrt(-e1 + np.sqrt(e1 ** 2 + e2 ** 2))
        # wse2
        wse2energy = 0.00000123982884337 / (wse2W * 1e-6)
        wse2_exp_params = np.array([3.55056955e+01, 4.78836239e+00,
                                    2.90640824e+00, 5.81521714e-01, 5.29399239e-01, 3.99176390e-01
                                       , 2.42574793e+00, 1.44599416e-01, 3.55051370e-01, 1.19803598e-12,
                                    # here
                                    1.7 + delta3, 2.16462602e-01, 0.007 + delta4,
                                    1.03755299e+00, 2.08861547e+00, 5.74162675e-02, 2.49299310e-01, 5.82112101e-11])

        er1_model_exp = tl_model.Lorentz_oscillator_model(wse2energy, *wse2_exp_params)
        e11 = er1_model_exp.real
        e22 = er1_model_exp.imag

        wse2N = (1 / np.sqrt(2)) * np.sqrt(e11 + np.sqrt(e11 ** 2 + e22 ** 2))
        wse2K = (1 / np.sqrt(2)) * np.sqrt(-e11 + np.sqrt(e11 ** 2 + e22 ** 2))

        # Interpolate to hBN wavelength grid
        wse2_n_i = np.interp(wavelengths, wse2W, wse2N)
        wse2_k_i = np.interp(wavelengths, wse2W, wse2K)
        ws2_n_i = np.interp(wavelengths, ws2W, ws2N)
        ws2_k_i = np.interp(wavelengths, ws2W, ws2K)
        si_n_i = np.interp(wavelengths, siW, siN)
        si_k_i = np.interp(wavelengths, siW, siK)
        sio2_n_i = np.interp(wavelengths, sio2W, sio2N)
        sio2_k_i = np.interp(wavelengths, sio2W, sio2K)
        c_n_i = np.interp(wavelengths, cW, cN)
        c_k_i = np.interp(wavelengths, cW, cK)

        # Stack into numpy array first
        nkvalues = np.column_stack((
            wavelengths,
            hbnN, wse2_n_i, ws2_n_i, c_n_i, sio2_n_i, si_n_i,
            hbnK, wse2_k_i, ws2_k_i, c_k_i, sio2_k_i, si_k_i
        ))

        # Filter wavelength range
        mask = (nkvalues[:, 0] >= 0.4) & (nkvalues[:, 0] <= 0.9)
        nkvalues_filtered = nkvalues[mask]

        # Convert to PyTorch tensor and move to device
        data_tensor = torch.tensor(nkvalues_filtered, dtype=torch.float64, device=self.device)

        return data_tensor

    def get_RC(self):
        """
        Calculate reflectance contrast using vectorized tensor operations.
        No loops - all wavelengths computed at once.
        Calculates BOTH sample and base in one pass for efficiency.
        """
        # Extract wavelengths and create complex refractive indices
        self.w = self.data[:, 0]
        wavelengths_m = self.w * 1e-6  # Convert to meters
        self.energy = 0.00000123982884337 / wavelengths_m

        # Complex refractive indices (all wavelengths at once)
        self.nair = torch.ones_like(self.w, dtype=torch.complex128)
        self.nhBN = torch.complex(self.data[:, 1], self.data[:, 7])
        self.nWSe2 = torch.complex(self.data[:, 2], self.data[:, 8])
        self.nWS2 = torch.complex(self.data[:, 3], self.data[:, 9])
        self.nC = torch.complex(self.data[:, 4], self.data[:, 10])
        self.nSiO2 = torch.complex(self.data[:, 5], self.data[:, 11])
        self.nSi = torch.complex(self.data[:, 6], self.data[:, 12])

        # Calculate both sample and base reflectance in one call
        rcSample, rcBase = self.calculate_reflectance()

        # Normalized contrast
        RC = (rcSample - rcBase) / rcBase

        return RC, self.energy

    def S(self, n1, n2, theta1=0):
        """
        Scattering matrix - vectorized for all wavelengths.
        """
        theta2 = torch.arcsin(n1 / n2 * torch.sin(torch.tensor(theta1, device=self.device)))

        r12 = ((n1 * torch.cos(torch.tensor(theta1, device=self.device))) -
               (n2 * torch.cos(theta2))) / \
              ((n1 * torch.cos(torch.tensor(theta1, device=self.device))) +
               (n2 * torch.cos(theta2)))
        t12 = 1 + r12
        r21 = -r12
        t21 = 1 + r21

        n_wl = n2.shape[0]
        S = torch.zeros(n_wl, 2, 2, dtype=torch.complex128, device=self.device)
        S[:, 0, 0] = t12
        S[:, 0, 1] = r21
        S[:, 1, 0] = r12
        S[:, 1, 1] = t21
        return S

    def MtoS(self, M):
        """Convert M to S matrix - vectorized."""
        A = M[:, 0, 0]
        B = M[:, 0, 1]
        C = M[:, 1, 0]
        D = M[:, 1, 1]

        S = torch.zeros_like(M)
        S[:, 0, 0] = (A * D - B * C) / D
        S[:, 0, 1] = B / D
        S[:, 1, 0] = -C / D
        S[:, 1, 1] = 1 / D
        return S

    def StoM(self, S):
        """Convert S to M matrix - vectorized."""
        t12 = S[:, 0, 0]
        r21 = S[:, 0, 1]
        r12 = S[:, 1, 0]
        t21 = S[:, 1, 1]

        M = torch.zeros_like(S)
        M[:, 0, 0] = (t12 * t21 - r12 * r21) / t21
        M[:, 0, 1] = r21 / t21
        M[:, 1, 0] = -r12 / t21
        M[:, 1, 1] = 1 / t21
        return M

    def sub(self, w, n, d, k):
        """Propagation matrix - vectorized."""
        w_m = w * 1e-6  # Convert to meters
        k0 = 2 * torch.pi / w_m
        exp = 1j * n * d * k0

        exp = torch.clamp(exp.real, -700, 700) + 1j * exp.imag

        a1 = torch.exp(-exp)
        a2 = torch.exp(exp)

        n_wl = w.shape[0]
        M = torch.zeros(n_wl, 2, 2, dtype=torch.complex128, device=self.device)
        M[:, 0, 0] = a1
        M[:, 1, 1] = a2
        return M

    def calculate_reflectance(self):
        """
        Calculate BOTH sample and base reflectance in one function.
        This is faster because we reuse many calculations.

        Returns:
            rcSample, rcBase - reflectance tensors
        """
        nm = 1e-9
        ang = 1e-10

        d1 = 1 * 3.348 * ang * self.d1
        d2 = self.d2 * nm
        d3 = 12.98 * ang / 2
        d4 = 12.167 * ang / 2
        d5 = self.d5 * nm
        d6 = 1 * 3.348 * ang * self.d6
        d7 = 90 * nm

        # Common layers (used in both sample and base)
        common_start = [
            (self.nair, self.nC, d1, self.data[:, 10]),
            (self.nC, self.nhBN, d2, self.data[:, 7]),
        ]

        # 2D material layers (only in sample)
        material_layers = [
            (self.nhBN, self.nWSe2, d3, self.data[:, 8]),
            (self.nWSe2, self.nWS2, d4, self.data[:, 9]),
        ]

        # Common end layers
        common_end = [
            (self.nWS2, self.nhBN, d5, self.data[:, 7]),  # For sample
            (self.nhBN, self.nC, d6, self.data[:, 10]),
            (self.nC, self.nSiO2, d7, self.data[:, 11]),
        ]

        # Base uses hBN instead of 2D materials
        base_middle = [
            (self.nhBN, self.nhBN, d5, self.data[:, 7]),
        ]

        # Build matrices for SAMPLE
        matrices_sample = []
        prev_n = self.nair

        for n1, n2, d, k in common_start:
            if d == 0:
                continue
            matrices_sample.append(self.StoM(self.S(prev_n, n2)))
            matrices_sample.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        for n1, n2, d, k in material_layers:
            if d == 0:
                continue
            matrices_sample.append(self.StoM(self.S(prev_n, n2)))
            matrices_sample.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        for n1, n2, d, k in common_end:
            if d == 0:
                continue
            matrices_sample.append(self.StoM(self.S(prev_n, n2)))
            matrices_sample.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        matrices_sample.append(self.StoM(self.S(prev_n, self.nSi)))

        # Calculate sample reflectance
        n_wl = self.w.shape[0]
        m = torch.eye(2, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(n_wl, -1, -1).clone()

        for M in matrices_sample:
            m = torch.bmm(M, m)

        ss = self.MtoS(m)
        rcSample = torch.abs(ss[:, 1, 0]) ** 2

        # Build matrices for BASE (reusing common_start calculations would be ideal)
        matrices_base = []
        prev_n = self.nair

        for n1, n2, d, k in common_start:
            if d == 0:
                continue
            matrices_base.append(self.StoM(self.S(prev_n, n2)))
            matrices_base.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        # Skip 2D materials, use base middle instead
        for n1, n2, d, k in base_middle:
            if d == 0:
                continue
            matrices_base.append(self.StoM(self.S(prev_n, n2)))
            matrices_base.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        # Adjust common_end for base (first layer connects differently)
        base_end = [
            (self.nhBN, self.nC, d6, self.data[:, 10]),
            (self.nC, self.nSiO2, d7, self.data[:, 11]),
        ]

        for n1, n2, d, k in base_end:
            if d == 0:
                continue
            matrices_base.append(self.StoM(self.S(prev_n, n2)))
            matrices_base.append(self.sub(self.w, n2, d, k))
            prev_n = n2

        matrices_base.append(self.StoM(self.S(prev_n, self.nSi)))

        # Calculate base reflectance
        m = torch.eye(2, dtype=torch.complex128, device=self.device).unsqueeze(0).expand(n_wl, -1, -1).clone()

        for M in matrices_base:
            m = torch.bmm(M, m)

        ss = self.MtoS(m)
        rcBase = torch.abs(ss[:, 1, 0]) ** 2

        return rcSample, rcBase

    def get_RC_with_deltas(self, delta1=0, delta2=0, delta3=0, delta4=0):
        """Get RC for specific delta values."""
        # Temporarily swap data
        old_data = self.data
        self.data = self.load_data(delta1, delta2, delta3, delta4)

        RC, eV = self.get_RC()

        # Restore original data
        self.data = old_data

        return RC, eV

    def get_RC_difference(self, noise=0.01,delta1=0, delta2=0, delta3=0, delta4=0):
        """Calculate RC(delta) - RC(0)."""
        RC_perturbed, eV = self.get_RC_with_deltas(delta1, delta2, delta3, delta4)
        RC_base, _ = self.get_RC()  # Uses self.data (delta=0)

        Y =  RC_perturbed - RC_base
        # Add noise to Y
        Y = Y + torch.randn_like(Y) * noise

        return Y, eV

    def graph1(self):
        """
        Graph RC without any delta shifts (base case).
        Includes peak detection, FWHM, and minima.
        """
        # Get base RC (no shifts)
        RC, eV = self.get_RC()

        # Convert to numpy for processing
        #RC = RC #+ torch.randn_like(RC) * 0.05
        RC_np = RC.cpu().numpy()
        eV_np = eV.cpu().numpy()

        # Find peaks (local maxima)
        peaks, peak_props = find_peaks(RC_np, prominence=0.01)

        # Find minima (local minima)
        minima, min_props = find_peaks(-RC_np, prominence=0.01)

        # Calculate FWHM for each peak
        fwhm_energies = []
        width_heights = []
        left_ips = []
        right_ips = []

        peaks = peaks[-2:]
        minima = minima[-2:]

        if len(peaks) > 0:
            widths, width_heights, left_ips, right_ips = peak_widths(RC_np, peaks, rel_height=0.5)
            # Convert widths from indices to energy units
            for i, peak_idx in enumerate(peaks):
                left_eV = np.interp(left_ips[i], np.arange(len(eV_np)), eV_np)
                right_eV = np.interp(right_ips[i], np.arange(len(eV_np)), eV_np)
                fwhm_energies.append(right_eV - left_eV)


        # Plot
        plt.figure(figsize=(12, 7))
        plt.plot(eV_np, RC_np, 'b-', linewidth=2, label='RC')

        # Plot peaks
        if len(peaks) > 0:
            plt.plot(eV_np[peaks], RC_np[peaks], 'ro', markersize=8, label='Peaks')

            # Annotate peaks with FWHM
            for i, peak_idx in enumerate(peaks):
                peak_eV = eV_np[peak_idx]
                peak_RC = RC_np[peak_idx]
                fwhm = fwhm_energies[i]

                plt.annotate(f'Peak: {peak_eV:.3f} eV\nFWHM: {fwhm:.4f} eV',
                             xy=(peak_eV, peak_RC),
                             xytext=(10, 10), textcoords='offset points',
                             fontsize=9, ha='left',
                             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                # Plot FWHM lines
                left_idx = int(left_ips[i])
                right_idx = int(right_ips[i])
                if left_idx < len(eV_np) and right_idx < len(eV_np):
                    plt.hlines(width_heights[i], eV_np[left_idx], eV_np[right_idx],
                               color='red', linestyle='--', alpha=0.5)

        # Plot minima
        if len(minima) > 0:
            plt.plot(eV_np[minima], RC_np[minima], 'gs', markersize=8, label='Minima')

            # Annotate minima
            for min_idx in minima:
                min_eV = eV_np[min_idx]
                min_RC = RC_np[min_idx]
                plt.annotate(f'Min: {min_eV:.3f} eV',
                             xy=(min_eV, min_RC),
                             xytext=(10, -20), textcoords='offset points',
                             fontsize=9, ha='left',
                             bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel('Reflectance Contrast', fontsize=12)
        plt.title('Reflectance Spectrum with Peak Analysis', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        #plt.show()
        return (RC_np[peaks[0]] - RC_np[minima[0]]) + 10 * (RC_np[peaks[1]] - RC_np[minima[1]])

    def extract_delta(self, Y, max_iter=15, epsilon=1e-6, learning_rate=1.0):
        """
        Extract delta parameters using Gauss-Newton optimization with PyTorch.
        Fits delta1, delta2, delta3, delta4 to match target Y = RC' - RC0

        Args:
            Y: Target difference (can be numpy array or tensor)
            max_iter: Maximum iterations
            epsilon: Small value to prevent division by zero
            learning_rate: Step size multiplier

        Returns:
            deltas: (delta1, delta2, delta3, delta4) as numpy array
        """
        # Convert Y to tensor
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(Y, dtype=torch.float64, device=self.device)

        # Initialize parameters (delta1, delta2, delta3, delta4)
        deltas = torch.zeros(4, dtype=torch.float64, device=self.device)

        # Get baseline RC (delta=0)
        RC0, eV = self.get_RC()

        # Finite difference step size
        eps = 1e-5

        print(f"Starting optimization...")

        for iteration in range(max_iter):
            # Current prediction
            RC_pred, _ = self.get_RC_with_deltas(
                deltas[0].item(), deltas[1].item(),
                deltas[2].item(), deltas[3].item()
            )

            # Residual: difference between target and prediction
            residual = Y - (RC_pred - RC0)

            # Weighted residual (inverse variance weighting)
            weights = torch.sqrt(1.0 / (torch.abs(residual) + epsilon))

            # Compute Jacobian using finite differences
            # J[i,j] = d(RC)/d(delta_j) at point i
            J = torch.zeros(len(RC0), 4, dtype=torch.float64, device=self.device)

            for j in range(4):
                # Perturb delta_j by +eps
                deltas_plus = deltas.clone()
                deltas_plus[j] += eps
                RC_plus, _ = self.get_RC_with_deltas(*deltas_plus.cpu().numpy())

                # Perturb delta_j by -eps
                deltas_minus = deltas.clone()
                deltas_minus[j] -= eps
                RC_minus, _ = self.get_RC_with_deltas(*deltas_minus.cpu().numpy())

                # Central difference
                J[:, j] = (RC_plus - RC_minus) / (2 * eps)

            # Weighted least squares: solve J^T W J * dp = J^T W * r
            W = torch.diag(weights)
            JW = J.T @ W  # (4, n_wavelengths)

            # Normal equations: (J^T W J) dp = J^T W r
            A = JW @ J  # (4, 4)
            b = JW @ residual  # (4,)

            # Solve for parameter update
            try:
                dp = torch.linalg.solve(A, b)
            except:
                # If singular, use pseudoinverse
                dp = torch.linalg.lstsq(A, b).solution

            # Update parameters
            deltas = deltas + learning_rate * dp

            # Check convergence
            step_size = torch.norm(dp).item()
            residual_norm = torch.norm(residual).item()

            if step_size < 1e-9:
                print(f"Converged after {iteration + 1} iterations")
                break

        return deltas.cpu().numpy()



# Usage:
if __name__ == "__main__":
    rc = ReflectanceCalculator(d1=1, d2=0, d5=0, d6=10)

    # Get difference
    dRC, eV = rc.get_RC_difference(0.05,0.0005,0.0005,0.0003,0.0003)

    stregnth = rc.graph1()

    plt.plot(eV.cpu().numpy(), dRC.cpu().numpy())
    plt.show()

    # Extract deltas
    deltas = rc.extract_delta(dRC, max_iter=20)

    print(f"Fitted deltas: {deltas}")

    # Verify fit
    RC_fitted, eV = rc.get_RC_with_deltas(*deltas)
