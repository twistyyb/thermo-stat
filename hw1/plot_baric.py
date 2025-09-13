from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


# Bryan Huang 9/13/2025

# HW 1 Q22:
# Consider a baric equation of state of the form
# P = N kBT/V + aT^2 e^(bPV)
# for some parameters a and b. It is not generally possible to solve for the isotherms in closed form,
# expressing P as a function of V or vice versa. Write a code that (i) inputs the relevant parameter values
# and variable range; (ii) given V , numerically solves for P (at fixed T ), within some specified precision,
# and then (iii) plots the isotherms. Do not just use a pre-packaged routine to graph level curves, but
# instead try out at least two of the following standard methods: direct functional iteration, Newton’s
# method (based on Taylor expansion), or bisection.
# Show some sample output for at least two different parameter values. Try to assess which of your
# methods is most efficient for your examples, in terms of number of floating point operators required to
# converge with the same (reasonably chosen) precision

# Constants
kB = 1.380649e-23 # Boltzmann constant
N = 6.022e23 # Number of particles
a = 1.0
b = 1.0e-4



T = [i for i in range(300, 1100, 100)]
V = 1 # m^3



# P(V, T) = N kBT/V + aT^2 e^(bPV)

def f(P, V, T):
    return P - N * kB * T / V - a * T**2 * np.exp(b * P * V)

def f_(P, V, T):
    return 1 - a * T ** 2 * b * V * np.exp(b * P * V)


threshold = 1.0e-6 # the tolerance for the root finding algorithm. stops if within threshold of root.
P_guess = 300 # Pa

# Set start and end points of V range.
V0 = 1.0e-3
V1 = 1.0e-2
points = 100
V_arr = np.linspace(V0, V1, points)

class Counter:
    def __init__(self):
        self.count = 0


# Newton implementation
# finds where f(p) = 0
# Uses the derivative of the function. Uses derivative f_val = df/dp, f(p) divided by df/dp gives the step dp to take towards root.
def newton_method(V, T, initial_guess, threshold, counter):
    p1 = initial_guess
    fval = f(p1, V, T)
    f_val = f_(p1, V, T)

    iter = 0
    while abs(fval) > threshold and iter < 10000:
        p0 = p1
        fval = f(p0, V, T)

        f_val = f_(p0, V, T)

        if f_val == 0:
            f_val = 0.0001
        p1 = p0 - fval / f_val 
        
        
        iter += 1
        # print(fval)
    counter.count += iter
    return p1


# Bisection implementation
# Determines an interval that captures a sign change, then takes midpoints, cutting interval in half until we narrow down to the root
def bisection_method(V, T, initial_guess, threshold, counter):
    p0 = 1.0e-6
    f0 = f(p0, V, T)
    p1 = initial_guess * 2

    # Find [p0, p1] such that f(p0) * f(p1) < 0, changes sign between p0 and p1
    search_iter = 0
    while f0 * f(p1, V, T) > 0:
        p1 *= 1.2
        search_iter += 1
    
    # print(f"Bisection: [p0, p1] = [{p0}, {p1}] (search_iter: {search_iter})")
    # print(f"f(p0) = {f0}, f(p1) = {f(p1, V, T)}")

    
    iter = 0
    while abs(f(p1, V, T)) > threshold and iter < 10000:
        midpoint = (p0 + p1) / 2
        fmidpoint = f(midpoint, V, T)
        if f0 * fmidpoint < 0: # if changes sign between p0 and midpoint, then midpoint is closer to the root than p1
            p1 = midpoint
        else: # if changes sign between midpoint and p1, then midpoint is closer to the root than p0
            p0 = midpoint
        iter += 1
        # keep narrowing down the interval to find the root
    # print("bisection required ", iter, " iterations")
    counter.count += iter
    




    counter.count += iter
    return p1






# Ideal gas implementation
def ideal_gas(V, T):
    return N * kB * T / V


counterNewton = Counter()
counterBisection = Counter()
newtonResults = {}
idealResults = {}
bisectionResults = {}


for T_sample in T:
    # Get Ideal gas results

    P2 = list(map(lambda v: ideal_gas(v, T_sample), V_arr))
    P2_arr = np.array(P2)
    idealResults[T_sample] = P2_arr

    # Get Newton results
    P = []
    for i, v in enumerate(V_arr):
        P_guess = P2[i]
        P.append(newton_method(v, T_sample, P_guess, threshold, counterNewton))
    P_arr = np.array(P)

    newtonResults[T_sample] = P_arr

    # Get Bisection results
    P = []
    for i, v in enumerate(V_arr):
        P_guess = P2[i]
        P.append(bisection_method(v, T_sample, P_guess, threshold, counterBisection))
    P_arr = np.array(P)

    bisectionResults[T_sample] = P_arr



    
    

print(f"newton iterations: {counterNewton.count} / {len(T) * points} points: {counterNewton.count / (len(T) * points)}")
percentDiff = np.mean(np.abs(np.array(list(newtonResults.values())) - np.array(list(idealResults.values())))) / np.mean(np.array(list(idealResults.values())))
print(f"Percent difference: {percentDiff*100:.2e} %")

print(f"bisection iterations: {counterBisection.count} / {len(T) * points} points: {counterBisection.count / (len(T) * points)}")
percentDiffBisection = np.mean(np.abs(np.array(list(bisectionResults.values())) - np.array(list(idealResults.values())))) / np.mean(np.array(list(idealResults.values())))
print(f"Percent difference bisection: {percentDiffBisection*100:.2e} %")





def Pa_to_MPa_arr(P_arr):
    return np.array(list(map(lambda x: x*1.0e-6, P_arr)))



fig: Figure
ax1: Axes
ax2: Axes

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.set_title(f"Newton Method ({points} samples/T) ({counterNewton.count / (len(T) * points)} iterations/sample)", size=8)
ax1.set_xlabel("V (m³)")
ax1.set_ylabel("P (MPa)")
for T_val, P_arr in list(newtonResults.items())[::-1]:
    ax1.plot(V_arr, Pa_to_MPa_arr(P_arr), label=f"{T_val} K")

ax2.set_title(f"Bisection Method ({points} samples/T) ({counterBisection.count / (len(T) * points)} iterations/sample)", size=8)
ax2.set_xlabel("V (m³)")
ax2.set_ylabel("P (MPa)")
for T_val, P_arr in list(bisectionResults.items())[::-1]:
    ax2.plot(V_arr, Pa_to_MPa_arr(P_arr), label=f"{T_val} K")


ax3.set_title(f"Ideal Comparison ({percentDiff*100:.2e} % difference)", size=8)
ax3.set_xlabel("V (m³)")
ax3.set_ylabel("P (MPa)")

for i, (T_val, P_arr) in enumerate(list(newtonResults.items())[::-1]):
    ax3.plot(V_arr, Pa_to_MPa_arr(P_arr), color="red", linewidth=1, label="Newton" if i == 0 else None)

for i, (T_val, P_arr) in enumerate(list(idealResults.items())[::-1]):
    ax3.plot(V_arr, Pa_to_MPa_arr(P_arr), color="blue", linewidth=1, label="Ideal" if i == 0 else None)



ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)

ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show(block=False)
input("Press Enter to continue...")

plt.savefig(f"./figs/baric_plot_a-{a}_b-{b}.png", dpi=300, bbox_inches="tight")
