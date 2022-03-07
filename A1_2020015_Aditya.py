import matplotlib.pyplot as plt
import numpy as np
import math

def N(u,a2,x):
    return (1/np.sqrt(2*math.pi*a2))*np.e**((-1/(2*a2))*((x-u)**2))

x = np.linspace(-5,8,1000)
plt.plot(x,N(2,1,x))
plt.xlabel('X')
plt.ylabel('P(x|w1)')
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('P(x|w1) vs x')
plt.show()

x = np.linspace(-4,10,1000)
plt.plot(x,N(5,1,x))
plt.xlabel('X')
plt.ylabel('P(x|w2)')
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('P(x|w2) vs x')
plt.show()

def N1(x):                 #returns N(2,1)/N(5,1)
    return  np.e**(0.5*(-6*x+21))

x = np.linspace(0,3,100)
plt.plot(x,N1(x))
plt.xlabel('X')
plt.ylabel('P(x|w1)/P(x|w2)')
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('P(x|w1)/P(x|w2) vs x')
plt.show()

def cauchy(a,b,x):
    return (1/(math.pi*b))*(1/(1+((x-a)/b)**2))

def f(a1,a2,b,X):
    return cauchy(a1,b,X)/(cauchy(a2,b,X)+cauchy(a1,b,X))   #Assuming P(w1) = P(w2)

x = np.linspace(-10,10,100)
plt.plot(x,f(3,5,1,x))
plt.xlabel('X')
plt.ylabel('P(w1|x)')
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('P(w1|x) vs x')
plt.show()