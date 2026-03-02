import numpy as np

def frequency_transfer(x_input, x_min=-1, x_max=1, f_exp_min=500, f_exp_max=1300):
    '''compute frequency in Hz from input using the ERB transfer function
    as implemented according to Glasberg & Moore, 1990, Eq. 4
    '''

    e_min = 21.4 * np.log10(4.37 * f_exp_min / 1000 + 1)
    e_max = 21.4 * np.log10(4.37 * f_exp_max / 1000 + 1)

    e = e_min + (e_max - e_min) * (x_input - (x_min)) / (x_max - (x_min))

    freq_out = (10 ** (e / 21.4) - 1) * (1000 / 4.37) 

    return freq_out



num = np.arange(-1, 1 + 0.01, 0.01)
hz = []

for n in num:
    f = frequency_transfer(n)
    hz.append(f)        
