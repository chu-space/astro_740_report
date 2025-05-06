import numpy as np
import matplotlib.pyplot as plt

matter_density = 2.775e11  # Mean matter density (h^2 M_sun / Mpc^3)
kronecker_c = 1.686   # Critical collapse overdensity
A, a = 0.3222, 0.707  # Sheth-Tormen model parameters

def mass_variance(M): # Mass viarance as described in the paper
    return 0.8 * (M / 1e13)**-0.3

def PS_mass_function(M): # Press-Schechter Mass Function
    variance = mass_variance(M)
    dln_variance_dlnM = -0.3  # Approximate constant derivative
    prefactor = np.sqrt(2 / np.pi) * (matter_density / M**2) * (kronecker_c / variance) * abs(dln_variance_dlnM)
    exponent = np.exp(-kronecker_c**2 / (2 * variance**2))
    return prefactor * exponent

def ST_mass_function(M): # Sheth-Tormen Mass Function
    variance = mass_variance(M)
    dln_variance_dlnM = -0.3
    prefactor = A * np.sqrt(2 * a / np.pi) * (matter_density / M**2) * (kronecker_c / variance)
    correction = (1 + (a * kronecker_c**2 / variance**2)**-0.3)
    exponent = np.exp(-a * kronecker_c**2 / (2 * variance**2))
    return prefactor * correction * exponent

M = np.logspace(10, 15, 100)  # Masses from 10^10 to 10^15 M_sun
n_PS = PS_mass_function(M)
n_ST = ST_mass_function(M)

n_values = [3, 4, 5] # n values insert to inline loop
H_n_asymptotic = [n**(n-2) for n in n_values]
bias_function = (M / 1e13)**0.1 # Mass bias function as described in paper

plt.figure(figsize=(8,6))
for i, n in enumerate(n_values):
    H_n = H_n_asymptotic[i] * (bias_function / np.max(bias_function)) + H_n_asymptotic[i]  # Toy model for H_n
    plt.loglog(M, H_n, label=f"H_{n}")

plt.xlabel(r"Halo Mass $M$ [$M_\odot$]", fontsize=15)
plt.ylabel(r"$H_n$", fontsize=15)
plt.title("Moments of Halo Distribution (Approximate)", fontsize=17)
plt.legend()
plt.tight_layout()
plt.show()

def NFW_function(r, rho_s, r_s): # Navvarow Frenk White mass function
    return rho_s / ((r / r_s) * (1 + r / r_s)**2)

# Parameters for a typical halo
r_s = 0.1  # Mpc
rho_s = 1e6  # M_sun/Mpc^3
r = np.logspace(-2, 1, 100)  # Radius from 0.01 Mpc to 10 Mpc
rho_nfw = NFW_function(r, rho_s, r_s) #Calculate NFW profile

plt.figure(figsize=(8,6))
plt.loglog(M, n_PS, label="Press-Schechter")
plt.loglog(M, n_ST, label="Sheth-Tormen", linestyle="--")
plt.xlabel(r"Halo Mass $M$ [$M_\odot$]", fontsize=15)
plt.ylabel(r"Mass Function $n(M)$ [h$^3$ Mpc$^{-3}$ $M_\odot^{-1}$]", fontsize=15)
plt.title("Halo Mass Functions", fontsize=17)
plt.legend()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.loglog(r, rho_nfw)
plt.xlabel(r"Radius $r$ [Mpc]", fontsize=15)
plt.ylabel(r"Density $\rho(r)$ [$M_\odot$ Mpc$^{-3}$]", fontsize=15)
plt.title("NFW Halo Density Profile", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()


