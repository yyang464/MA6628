# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:13:26 2018

@author: Admin
"""

#********************** Black-Scholes-Merton (1973) European Call & Put Valuation*******************

# 05_com/BSM_option_valuation.py
# Derivatives Analytics with Python

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def d1f(St, K, t, T, r, sigma):
    ''' Black-Scholes-Merton d1 function.
        Parameters see e.g. BSM_call_value function. '''
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * math.sqrt(T - t))
    return d1


# Valuation Functions
def BSM_call_value(St, K, t, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European call option value.
    
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
        
    Returns
    =======
    call_value : float
        European call present value at t
    '''
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * ss.norm.cdf(d1) - math.exp(-r * (T - t)) * K * ss.norm.cdf(d2)
    return call_value





def BSM_put_value(St, K, t, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European put option value.
    
    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        date of maturity/time-to-maturity if t = 0; T > t
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
        
    Returns
    =======
    put_value : float
        European put present value at t
    '''


    put_value = BSM_call_value(St, K, t, T, r, sigma) \
        - St + math.exp(-r * (T - t)) * K
    return put_value




#******************************************CRR***********************************************
    
# Cox-Ross-Rubinstein Binomial Model
# European Option Valuation

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
from BSM_option_valuation import BSM_call_value


# Model Parameters
S0 = 100.0  # index level
K = 100.0  # option strike
T = 1.0  # maturity date
r = 0.05  # risk-less short rate
sigma = 0.2  # volatility


# Valuation Function
def CRR_option_value(S0, K, T, r, sigma, otype, M=4):    
    ''' Cox-Ross-Rubinstein European option valuation.
    Parameters
    ==========

    S0 : float
        stock/index level at time 0
    K : float
        strike price
    T : float
        date of maturity
    r : float
        constant, risk-less short rate
    sigma : float
        volatility
    otype : string
        either 'call' or 'put'
    M : int
        number of time intervals
    '''

    # Time Parameters
    dt = T / M  # length of time interval
    df = math.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = math.exp(sigma * math.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (math.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md


    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option

    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1] +
                         (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]


def plot_convergence(mmin, mmax, step_size):
    ''' Plots the CRR option values for increasing number of time
    intervals M against the Black-Scholes-Merton benchmark value.'''
    BSM_benchmark = BSM_call_value(S0, K, 0, T, r, sigma)
    m = range(mmin, mmax, step_size)
    CRR_values = [CRR_option_value(S0, K, T, r, sigma, 'call', M) for M in m]
    plt.figure(figsize=(9, 5))
    plt.plot(m, CRR_values, label='CRR values')
    plt.axhline(BSM_benchmark, color='r', ls='dashed', lw=1.5,
                label='BSM benchmark')
    plt.xlabel('# of binomial steps $M$')
    plt.ylabel('European call option value')
    plt.legend(loc=4)
    plt.xlim(0, mmax)



#***************************************implied volatility*****************************************

from scipy import optimize
from BSM_option_valuation import *

def IVolBsm(S0, K, T, r, P0):
    """
    Inputs:
        S0: spot price
        K: strike
        T: time to maturity in year
        r: rate
        P0: market price
    Outputs:
        Implied volatility
    """
    InitVol = .3
    error = lambda sigma: (BSM_call_value(S0, K, 0, T, r, sigma) - P0)**2
    opt = optimize.fmin(error, InitVol);
    return opt[0]

if __name__ == "__main__":
    print('Implied volatility is', IVolBsm(100, 100, 1, .02, 9))



#*******************************说明文件*******************************************
'''
L01s01.ipynb        Python Fundamentals
L02s01.ipynb        Basic Probability Review
L03s01.ipynb        Basic Monte Carlo
#蒙特卡罗方法，又称随机模拟方法或统计模拟方法，是在20世纪40年代随着电子计算机的发明而提出的。
它是以统计抽样理论为基础，利用随机数，经过对随机变量已有数据的统计进行抽样实验或随机模拟，
以求得统计量的某个数字特征并将其作为待解决问题的数值解。
蒙特卡洛模拟方法的基本原理是：假定随机变量X1、X2、X3……Xn、Y，其中X1、X2、X3……Xn的概率分布已知，
且X1、X2、X3……Xn、Y有函数关系：Y=F（X1、X2、X3……Xn），希望求得随机变量Y的近似分布情况及数字特征。
通过抽取符合其概率分布的随机数列X1、X2、X3…Xn带入其函数关系式计算获得Y的值。当模拟的次数足够多的时候，
我们就可以得到与实际情况相近的函数Y的概率分布和数字特征。
蒙特卡洛法的特点是预测结果给出了预测值的最大值，最小值和最可能值，给出了预测值的区间范围及分布规律。

L03s02.ipynb       Sampling: Inverse transform method
L04s01.ipynb       Black and Scholes formula
L04s02.ipynb       Calibration of BSM Volatility
The general approach is that, one defines an error function that is to be minimized.
RMSE as an error function
'''
import numpy as np
import pandas as pd
import scipy.optimize as sop

def generate_plot(opt, options):
    #
    # Calculating Model Prices
    #
    sigma = opt
    options['Model'] = 0.0
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        options.loc[row, 'Model'] = BSM_call_value(S0, option['Strike'], 0, T, r, sigma)

    #
    # Plotting
    #
    mats = sorted(set(options['Maturity']))
    options = options.set_index('Strike')
    for i, mat in enumerate(mats):
        options[options['Maturity'] == mat][['Call', 'Model']].\
            plot(style=['b-', 'ro'], title='%s' % str(mat)[:10],
                 grid=True)
        plt.ylabel('option value')
        plt.savefig('BSM_calibration_3_%s.pdf' % i)
     #The following graph shows that how the market data differs 
     #from the theoretical price by just making a guess of  volatility.   
generate_plot(.3, options); #Arbitrary input for the volatility

def BSM_error_function(p0):
    ''' Error Function for parameter calibration in BSM Model via
    
    Parameters
    ==========
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    RMSE: float
        root mean squared error
    '''
    global i, min_RMSE
    sigma = p0
    if sigma < 0.0:
        return 500.0
    se = []
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        model_value = BSM_call_value(S0, option['Strike'], 0, T, r, sigma)
        se.append((model_value - option['Call']) ** 2)
    RMSE = math.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    if i % 5 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    i += 1
    return RMSE

''' L05s01.ipynb       Approximation to BSM Option Valuation by CRR
CRR Option Pricing
In 1979, Cox, Ross and Rubinstein presented (cf. Cox et al. (1979)) 
their binomial option pricing model. This model assumes in principle a BSM economy 
but in discrete time with discrete state space. 
Whereas the BSM model necessitates advanced mathematics and the handling of partial 
differential equations (PDE), the CRR analysis relies on fundamental probabilistic concepts only. 
Their representation of uncertainty by binomial (recombining) trees is still today the tool of 
choice when one wishes to illustrate option topics in a simple, intuitive way. 
Furthermore, their numerical approach allows not only European options but also American options 
to be valued quite as easily.

M76_valuation_FFT.py   M76 Characteristic Function
'''

def M76_characteristic_function(u, x0, T, r, sigma, lamb, mu, delta):

    ''' Valuation of European call option in M76 model via

    Lewis (2001) Fourier-based approach: characteristic function.



    Parameter definitions see function M76_value_call_INT. '''

    omega = x0 / T + r - 0.5 * sigma ** 2 \

        - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)

    value = np.exp((1j * u * omega - 0.5 * u ** 2 * sigma ** 2 +

                    lamb * (np.exp(1j * u * mu -

                    u ** 2 * delta ** 2 * 0.5) - 1)) * T)

    return value



#

# Valuation by FFT

#





def M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta):

    ''' Valuation of European call option in M76 model via

    Carr-Madan (1999) Fourier-based approach.



    Parameters

    ==========

    S0: float

        initial stock/index level

    K: float

        strike price

    T: float

        time-to-maturity (for t=0)

    r: float

        constant risk-free short rate

    sigma: float

        volatility factor in diffusion term

    lamb: float

        jump intensity

    mu: float

        expected jump size

    delta: float

        standard deviation of jump



    Returns

    =======

    call_value: float

        European call option present value

    '''

    k = math.log(K / S0)

    x0 = math.log(S0 / S0)

    g = 2  # factor to increase accuracy

    N = g * 4096

    eps = (g * 150.) ** -1

    eta = 2 * math.pi / (N * eps)

    b = 0.5 * N * eps - k

    u = np.arange(1, N + 1, 1)

    vo = eta * (u - 1)

    # Modificatons to Ensure Integrability

    if S0 >= 0.95 * K:  # ITM case

        alpha = 1.5

        v = vo - (alpha + 1) * 1j

        mod_char_fun = math.exp(-r * T) * M76_characteristic_function(

            v, x0, T, r, sigma, lamb, mu, delta) \

            / (alpha ** 2 + alpha - vo ** 2 + 1j * (2 * alpha + 1) * vo)

    else:  # OTM case

        alpha = 1.1

        v = (vo - 1j * alpha) - 1j

        mod_char_fun_1 = math.exp(-r * T) * (1 / (1 + 1j * (vo - 1j * alpha))

                                             - math.exp(r * T) /

                                             (1j * (vo - 1j * alpha))

                                             - M76_characteristic_function(

                    v, x0, T, r, sigma, lamb, mu, delta) /

                    ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha)))

        v = (vo + 1j * alpha) - 1j

        mod_char_fun_2 = math.exp(-r * T) * (1 / (1 + 1j * (vo + 1j * alpha))

                                             - math.exp(r * T) /

                                             (1j * (vo + 1j * alpha))

                                             - M76_characteristic_function(

                    v, x0, T, r, sigma, lamb, mu, delta) /

                    ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha)))



    # Numerical FFT Routine

    delt = np.zeros(N, dtype=np.float)

    delt[0] = 1

    j = np.arange(1, N + 1, 1)

    SimpsonW = (3 + (-1) ** j - delt) / 3

    if S0 >= 0.95 * K:

        fft_func = np.exp(1j * b * vo) * mod_char_fun * eta * SimpsonW

        payoff = (fft(fft_func)).real

        call_value_m = np.exp(-alpha * k) / math.pi * payoff

    else:

        fft_func = (np.exp(1j * b * vo) *

                    (mod_char_fun_1 - mod_char_fun_2) *

                    0.5 * eta * SimpsonW)

        payoff = (fft(fft_func)).real

        call_value_m = payoff / (np.sinh(alpha * k) * math.pi)

    pos = int((k + b) / eps)

    call_value = call_value_m[pos]

    return call_value * S0





if __name__ == '__main__':

    print("Value of Call Option %8.3f"

        % M76_value_call_FFT(S0, K, T, r, sigma, lamb, mu, delta))