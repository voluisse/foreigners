import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- True PES data (from Morse-type approximation above) ---
distances = np.array([1.60, 1.80, 2.00, 2.20, 2.40,
                      2.60, 2.70, 2.80, 3.00, 3.50,
                      4.00, 5.00])
energies_true = np.array([15.7422, 6.7177, 1.7786, -0.7702, -1.9482,
                          -2.3631, -2.4030, -2.3728, -2.1846, -1.4892,
                          -0.9048, -0.2949])

# --- Morse potential ---
def morse(r, D, a, r_e):
    return D * (1 - np.exp(-a * (r - r_e)))**2 - D

# --- Initial guess ---
p0 = [1.0,1.0,1.0]  # D, a, r_e

# Compute curve for initial guess
energies_initial = morse(distances, *p0)

# --- todostart ---
D_grid=np.linspace(1.0,4.0,61)
a_grid=np.linspace(0.3,2.0,86)
re_grid=np.linspace(2.2,3.0,81)

def mse(yhat,y):
  return np.mean((yhat-y)**2)

best_error=np.inf
best_params=None
D_best=a_best=re_best=None

for D in D_grid:
  for a in a_grid:
    for re in re_grid:
      E_pred=morse(distances,D,a,re)
      err=mse(E_pred,energies_true)
      if err<best_error:
        best_error=err
        D_best,a_best,re_best=D,a,re
        best_params=(D_best,a_best,re_best)

# --- todoend ---
                   
energies_best = morse(distances, *best_params)

print("Best Morse parameters from grid search:")
print(f"  D_e = {D_best:.4f} eV")
print(f"  a   = {a_best:.4f} Å⁻¹")
print(f"  r_e = {re_best:.4f} Å")
print(f"  Error = {best_error:.6f}")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

# Plot 1: Initial guess
ax1.plot(distances, energies_true, 'o-', label="True PES (data)", color="black")
ax1.plot(distances, energies_initial, 's--',
         label=f"Initial Morse Guess (D={p0[0]:.2f}, a={p0[1]:.2f}, rₑ={p0[2]:.2f})",
         color="red")
ax1.axhline(0, color="gray", linestyle=":")
ax1.set_ylabel("Energy (eV)", fontsize=12)
ax1.set_title("Al₂ Dimer PES: Initial Guess vs True")
ax1.legend()

# Plot 2: Best fit from grid search
ax2.plot(distances, energies_true, 'o-', label="True PES (data)", color="black")
ax2.plot(distances, energies_best, 's--',
         label=f"Best Morse Fit (D={D_best:.2f}, a={a_best:.2f}, rₑ={re_best:.2f})",
         color="blue")
ax2.axhline(0, color="gray", linestyle=":")
ax2.set_xlabel("Bond distance r (Å)", fontsize=12)
ax2.set_ylabel("Energy (eV)", fontsize=12)
ax2.set_title("Best Grid Search Fit vs True")
ax2.legend()

plt.tight_layout()
plt.show()
