import random as rand
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import eig

def MLE(samples,n):
    mle = [0,0]
    for i in range(n):
        mle[0] = mle[0] + samples[i][0]
        mle[1] = mle[1] + samples[i][1]
    mle[0] = mle[0]/n
    mle[1] = mle[1]/n
    return mle    

def assign_values(dim, num,Samples):
    i = 0
    while (i < num):
        index = rand.randrange(0, 100)
        if (Samples[index][dim] == 1):
            continue
        Samples[index][dim] = 1
        i = i + 1

def generate_samples(u,samples):
    assign_values(0,u[0],samples)
    assign_values(1,u[1],samples)
    
def g(theta,sample):
    ans = 0
    for i in range(2):
        ans = ans + sample[i]*math.log(theta[i]) + (1-sample[i])*math.log(1-theta[i]) + math.log(theta[i])
    return ans    
   
# part a    
numRows = 100
numCols = 2
Class1Samples = np.zeros(shape=(numRows ,numCols), dtype=np.uint8)
Class2Samples = np.zeros(shape=(numRows ,numCols), dtype=np.uint8)
u1x100 = [50,80]
u2x100 = [90,20]
generate_samples(u1x100,Class1Samples)
generate_samples(u2x100,Class2Samples)
print("class 1 samples = \n" , Class1Samples)
print("class 2 samples = \n" , Class2Samples)

# part b
u1 = MLE(Class1Samples,50)
print("Class Conditional Parameter of training set of Class 1 Samples  = ", u1)
x = [0]*50
y = [[0]*2]*50
for i in range(50):
    x[i] = i+1
    y[i] = MLE(Class1Samples,i+1) 

plt.plot(x,y)
plt.xlabel("n values")
plt.ylabel("MLE of Class 1 samples")
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('MLE vs n Plot')
plt.show()

# part c
u2 = MLE(Class2Samples,50)
print("Class Conditional Parameter of training set of Class 2 Samples  = ", u2)
for i in range(50):
    y[i] = MLE(Class2Samples,i+1) 
plt.plot(x,y)    
plt.xlabel("n values")
plt.ylabel("MLE of Class 2 samples")
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('MLE vs n Plot')
plt.show() 

# part d
z = [0]*50
y = [0]*50
a = [0]*50
b = [0]*50
for i in range(50):
    z[i] = Class1Samples[i][1]
    y[i] = Class1Samples[i][0]
    a[i] = Class2Samples[i][0]
    b[i] = Class2Samples[i][1]
plt.title("Scatter Plot of class 1 samples")    
plt.xlabel("n values")
plt.ylabel("dimension 1")
plt.scatter(x,y)
plt.show()
plt.title("Scatter Plot of class 1 samples") 
plt.xlabel("n values")
plt.ylabel("dimension 2")
plt.scatter(x,z)
plt.show()

plt.title("Scatter Plot of class 2 samples")    
plt.xlabel("n values")
plt.ylabel("dimension 1")
plt.scatter(x,a)
plt.show()
plt.title("Scatter Plot of class 2 samples") 
plt.xlabel("n values")
plt.ylabel("dimension 2")
plt.scatter(x,b)
plt.show()

# part e
count = [0,0]
for i in range(50,len(Class1Samples)):
    if (g(u1,Class1Samples[i]) > g(u2,Class1Samples[i])):
        count[0] = count[0] + 1
    if (g(u2,Class2Samples[i]) > g(u1,Class2Samples[i])):  
        count[1] = count[1] + 1  
print("Number of samples Correctly classified for test set 1 = " , count[0])   
print("Number of samples Correctly classified for test set 2 = " , count[1])       


# question 3
def transpose(x):
    y = x[0][1] 
    x[0][1] = x[1][0]
    x[1][0] = y
    return x

# PCA
N = 2
X = np.array([[1,5],[2,4]])
print("X = \n", X)
M = np.mean(X.T, axis=0)
Xc = X - M
print("Centralised(X) = \n", Xc)
S = np.dot(Xc, Xc)
values, vectors = eig(S)
print("PCA Matrix = \n" , vectors)
print("EigenValues = " ,values)  
Ut = transpose(vectors)
Y = np.dot(Ut,Xc)
print("Encoding Matrix Y = \n" , Y)
U = transpose(vectors)

# MSE
A = np.dot(U,Y) + M
array = np.subtract(A, X)  
squared_array = np.square(array)
mse = squared_array.mean()
print("MSE = ",mse)  

# question 3 part d
def f(n):
    array = np.zeros(shape=(D ,n))
    for i in range(n):
        for j in range(D):
            array[j][i] = vectors[j][i]
    Y = np.dot(np.transpose(vectors),X)  
    A = np.dot(np.transpose(vectors),Y)
    centralization(A,Ux,n,1)
    array = np.subtract(A, X)  
    squared_array = np.square(array)
    return squared_array.mean()    

def centralization(X,u,n,val):
    for i in range(D):
        for j in range(n):
            if (val == 0):
                X[i][j] = X[i][j] - u[i]
            else:
                X[i][j] = X[i][j] + u[i]    

N = 10            # Change the number of samples here
D = 5             # Change the number of dimensions here
X = np.zeros(shape=(D ,N))
for i in range(D):
    for j in range(N):
        X[i][j] = round(rand.random(),2)     
Ux = [0]*D
for i in range(D):
    for j in range(N):
        Ux[i] = Ux[i] + X[i][j] / N 
print("Mean Vector(Ux) = \n",Ux)        
cov_array = np.cov(X)
print("covariance Matrix = \n",cov_array)
print("Data Matrix = \n",X)  
values, vectors = eig(cov_array)
print("U Matrix = \n",vectors)
centralization(X,Ux,N,0)
print("MSE = ",f(D))  

samples = []
y = []
for i in range(1,D+1):
    samples.append(D+1-i)
    y.append(f(i))  
plt.plot(samples,y)
plt.xlabel("n Values")
plt.ylabel("MSE")
plt.axhline(color = 'black')
plt.axvline(color = 'black')
plt.title('MSE vs n Plot')
plt.show()   