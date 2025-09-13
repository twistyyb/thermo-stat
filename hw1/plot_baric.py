import numpy as np
import matplotlib.pyplot as plt


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
a = 1.0e-10
b = 1.0e-5



T = 300
V = 1 # m^3



# P(V, T) = N kBT/V + aT^2 e^(bPV)

def f(P, V, T):
    return P - N * kB * T / V - a * T**2 * np.exp(b * P * V)

def f_(P, V, T):
    return 1 - a * T ** 2 * b * V * np.exp(b * P * V)


convergence = 1.0e-10 # diff less than 1 Pa
P_guess = 300 # Pa


V0 = 1.0e-3
V1 = 1.0e-1
points = 10
V = np.linspace(V0, V1, points)

def newton_method(V, T, initial_guess, diff_threshold):
    p1 = initial_guess
    fval = f(p1, V, T)
    f_val = f_(p1, V, T)

    count = 0

    while fval > diff_threshold:
        p0 = p1

        fval = f(p0, V, T)
        f_val = f_(p0, V, T)

        if f_val == 0:
            f_val == 0.0001
        p1 = p0 - fval / f_val 
        print(f"V: {V}, T: {T} => P:{p1}")
        count += 1
    print(f"V: {V}, T: {T} => P:{p1}")
    input()
    return p1


P = []  # Start with empty list
for v in V:
    P.append(newton_method(v, 300, P_guess, convergence))

print(P)
P = np.array(P)  # Convert to numpy array for plotting

print(V)
print(P)



