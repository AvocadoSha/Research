#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import argparse
import matplotlib.pyplot as plt
import os

def co2_model(x, y0, A, k):
    return y0 + A * (1 - np.exp(-k * x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit rate constant for CO2 evolution from x-y data file'
    )
    parser.add_argument('filename', type=str, help='Filename of the x-y data file')
    args = parser.parse_args()

    # Load data skipping header line
    data = np.loadtxt(args.filename, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # Initial guess
    y0_guess = y[0]
    A_guess = y[-1] - y0_guess if (y[-1] > y0_guess) else max(y) - y0_guess
    if A_guess <= 0:
        A_guess = max(y)
    k_guess = 0.1

    initial_guess = [y0_guess, A_guess, k_guess]

    # Fit
    popt, pcov = curve_fit(co2_model, x, y, p0=initial_guess)
    y0_fit, A_fit, k_fit = popt

    y_fit = co2_model(x, *popt)

    # Save fitted results: time, observed, fitted
    output_filename = f"fitted_{os.path.splitext(os.path.basename(args.filename))[0]}.dat"
    np.savetxt(output_filename, np.column_stack((x, y, y_fit)),
               header="time(obs)\tCO2(obs)\tCO2(fitted)", fmt="%.6f")

    # Plot
    plt.scatter(x, y, label='Experimental data')
    plt.plot(x, y_fit, 'r-', label=f'Fit: y0={y0_fit:.2f}, A={A_fit:.2f}, k={k_fit:.4f}')
    plt.xlabel('Time')
    plt.ylabel('CO2 molecules')
    plt.legend()
    plt.title('Non-linear fitting of CO2 evolution')
    plt.savefig("fitted_plot.pdf", format="pdf")
    plt.show()

    print(f'Fitted parameters:')
    print(f'  Baseline y0 = {y0_fit:.2f} CO2 molecules')
    print(f'  Max increment A = {A_fit:.2f} CO2 molecules')
    print(f'  Rate constant k = {k_fit:.4f} (1/time unit)')
    print(f'Fitted data saved to: {output_filename}')

