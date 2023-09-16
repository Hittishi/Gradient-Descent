import numpy as np
from matplotlib import pyplot as plt

def Random_Q(d) :

    A = []
    for i in range (d):

        A.append (np.random.standard_normal (size =d)) #aD×DmatrixA, where every column is a vector TO GENERATE 0
    Q= np.matmul (np. transpose (A), A)
    return Q

def Random_c (d) :

    c = np.random.standard_normal(size=(d, 1)) #D-dimensional vector where each component is drawn from a standard normaldistribution.

    return c

def find_pk(Q, c,x): # To find out derivate of the Function F
    p = np.subtract (np.matmul (Q,x), c)
    return p

def find_qk(x,x_prev): # To find q at every Kth iteration
    qk= np.subtract (x,x_prev)
    return qk

def find_F(x,Q,c): # computing the value of the function
    x_t = np.transpose (x)
    G= (0.5* (np.matmul (x_t, np.matmul (Q, x))))
    H = np.matmul (x_t, c)
    F= np.subtract (G, H)
    return F.item()

def find_alpha (pk, Q) : # Generating the alpha for optimized Gradient Descent at every Kth iteration

    pk_t = np.transpose(pk)
    x = np.matmul (pk_t, pk)
    y = np.matmul (pk_t, np.matmul (Q, pk))
    alpha = x / y
    alpha = x / y
    return alpha.item ()

def Find_xmin(Q,c): # find x* = 0-1c
    xmin = np.matmul (np.linalg.inv (Q), c)
    return xmin

def find_aoa(xnew, x, xmin): # find anlgle btween the minimizer x* and X(k) at every Kth iteration
    x= np.subtract (xnew, x) 
    y= np.subtract (xmin, x)
    norm1 = np.linalg.norm (x)
    norm2 = np.linalg.norm(y)
    numerator= np.matmul (np.transpose (x),y)
    aoa = (numerator.item())/ (norm1*norm2)
    return aoa

def find_alpha_beeta(pk, qk,Q) : #find alpha and beta for optimized gradient descent with momentum for every kth iteration
    pk_t = np.transpose (pk)
    qk_t = np.transpose (qk)
    pk_t_pk = np.matmul (pk_t,pk)
    qk_t_pk = np.matmul (qk_t,pk)
    pk_t_Q_qk = np.matmul(pk_t,np.matmul (Q,qk))
    qk_t_Q_qk = np.matmul(qk_t,np.matmul (Q, qk))
    pk_t_Q_pk = np.matmul(pk_t, np.matmul (Q,pk))
    x = np.subtract(np.matmul(pk_t_pk,qk_t_Q_qk),np.matmul(qk_t_pk,pk_t_Q_qk))
    y = np.subtract (np.matmul (pk_t_Q_pk,qk_t_Q_qk),np.matmul (pk_t_Q_qk,pk_t_Q_qk))
    alpha = x/y
    a= np.subtract (np.matmul (pk_t_pk,pk_t_Q_qk),np.matmul (qk_t_pk,pk_t_Q_pk))
    b= np.subtract(np.matmul(qk_t_Q_qk,pk_t_Q_pk),np.matmul (pk_t_Q_qk,pk_t_Q_qk))
    beeta = a/b
    return alpha.item(), beeta.item()

def find_qko(pk) : #To generate an orthogonal vector to pk for orthogonal optimized gradient Descent with momentum
    x= Random_c(10)
    x_pk = np.matmul (np.transpose (pk), x)
    pk_pk = np.matmul (np.transpose (pk) ,pk)
    temp = x_pk/pk_pk
    qk_o = np.subtract(x,np.dot(temp.item(),pk))
    return qk_o

def error (x, xmin): # find the error between X* ( minimizer) and computed × at Kth iteration
    error = np. subtract (x, xmin)
    error = np.linalg.norm(error)
    return error

def optimized_gradDesc(x, Q,c): # Performing Grandient Descent with optimized alpha

    iteration = 0
    e_arr = []
    aoa_arr = []
    L_arr = []
    xmin = Find_xmin(Q, c)
    pk = find_pk(Q, c, x)
    F = find_F(x, Q, c)
    alpha = find_alpha (pk, Q)
    e = error (x, xmin)
    while (e > 0.001) :
        xnew = np.subtract (x, np. dot (alpha, pk))
        e = error (xnew, xmin)
        e_arr.append(e)
        aoa = find_aoa (xnew,x, xmin)
        aoa_arr.append (aoa)
        pk = find_pk(Q, c, xnew)
        F = find_F(xnew, Q, c)
        L_arr. append (F)
        alpha = find_alpha (pk, Q)
        x = xnew
        iteration = iteration + 1
    print ("OPTIMIZED GRADIENT DESCENT")
    print("No of iterations to reach mimnimum in", iteration)
    print("Minimum obtained by optimized gradient descent",F)
    print("Error obtained",e)
    return aoa_arr, e_arr,L_arr

def gradient_descent (x, Q,c): #performing vanilla gradient descent with constant step size
    aoa_arr = []
    e_arr = []
    L_arr = []
    iteration=0
    xmin = Find_xmin (Q, c)
    pk = find_pk(Q, c, x)
    F = find_F(x, Q, c)
    alpha = 0.01
    e = error (x, xmin)
    while (e > 0.001):
        xnew = np.subtract (x, np. dot (alpha, pk))
        e = error (xnew, xmin)
        e_arr.append(e)
        aoa = find_aoa (xnew, x, xmin)
        aoa_arr.append (aoa)
        pk = find_pk(Q, c, xnew)
        F = find_F(xnew, Q, c)
        L_arr. append (F)
        iteration = iteration + 1
    print ("VANILLA GRADIENT DESCENT")
    print ("No of iterations to reach mimnimum in ",iteration)
    print ("Minimum obtained ", F)
    print("Error obtained ",e)
    return aoa_arr, e_arr, L_arr



def momentum(x, Q,c): # Gradient Descent with constant alpha and beta

    L_arr = []

    x_prev = 0
    e_arr = []
    aoa_arr = []
    xmin = Find_xmin (Q, c)
    pk = find_pk(Q, c, x)
    qk = find_qk(x, x_prev)
    F = find_F(x, Q, c)
    alpha = 0.01
    beeta = 0.1
    iteration = 0
    e= error (x,xmin)
    while (e > 0.001):
        xnew = np.add(np.subtract(x, np. dot (alpha,
        pk)), np.dot (beeta, qk))
        e = error (xnew, xmin)
        e_arr.append(e)
        aoa = find_aoa (xnew, x, xmin)
        aoa_arr.append (aoa)
        x_prev = x
        x = xnew
        pk = find_pk(Q, c, x)
        L_arr. append (F)
        iteration = iteration + 1
    print ("MOMENTUM GRADIENT DESCENT")
    print ("No of iterations to reach minimum in iteration",iteration)
    print ("Minimum obtained by Momentum Gradient Descent", F)
    print("Error obtained ", e)
    return aoa_arr,e_arr, L_arr

def optimizedMomentum(x, Q,c) : # Performing Gradient descent with optimized beta and alpha
    L_arr = []
    aoa_arr = []
    x_prev = 0
    e_arr = []
    xmin = Find_xmin(0, c)
    pk = find_pk(Q, c, x)
    qk = find_qk(x, x_prev)
    F = find_F(x, Q, c)
    alpha, beeta = find_alpha_beeta(pk, qk, Q)
    iteration = 0
    e= error(x,xmin)
    while (e > 0.001):
        xnew = np.add(np.subtract (x, np.dot (alpha,pk)), np.dot (beeta, qk))
        e = error (xnew, xmin)
        e_arr.append(e)
        aoa = find_aoa (xnew, x, xmin)
        aoa_arr.append (aoa)
        x_prev = x
        x = xnew
        pk = find_pk(Q, c, x)
        qk = find_qk(x, x_prev)
        alpha, beeta= find_alpha_beeta(pk, qk, Q)
        F = find_F(x, Q, c)
        L_arr.append (F)
        iteration = iteration + 1
    print ("OPTIMIZED MOMENTUM GRADIENT DESCENT")
    print ("Minimun of the Function:" , find_F(xmin, Q(c)))
    print("No of iterations to reach minimum in ", iteration)
    print("Minimum obtained by Optimized Momentum Gradient Descent", F)
    print("Error abtained ", e)

    return aoa_arr, e_arr, L_arr

def orthogonalMomentum (x, Q,c): #Gradient Descent by producing an additional vector that is orthogonal to pk for every iteration
    L_arr = []
    x_prev = 0
    e_arr = []
    aoa_arr = []
    xmin = Find_xmin (0, c)
    pk = find_pk(Q, c, x)
    qk_o= find_qko (pk)
    F = find_F(x, Q, c)
    alpha, beeta = find_alpha_beeta(pk, qk_o,Q)
    iteration = 0
    e= error (x,xmin)
    while (e > 0.001):
        xnew = np.add(np.subtract(x, np. dot (alpha,pk)), np.dot (beeta, qk_o))
        e = error (xnew, xmin)
        e_arr.append (e)
        aoa = find_aoa(xnew, x, xmin)
        aoa_arr.append(aoa)
        X_prev = x
        X = xnew
        pk = find_pk(Q, c, x)
        qk_o = find_qko (pk)
        alpha, beeta= find_alpha_beeta(pk, qk_o, Q)
        F = find_F(x, Q, c)
        L_arr.append (F)
        iteration - iteration + 1
    print ("OPTIMIZED ORTHOGONAL MOMENTUM GRADIENT DESCENT")
    print ("No of iterations to reach minimum in ",iteration)
    print ("Minimum obtained by Optimized Orthogonal Momentum Gradient Descent",F)
    print ("Error obtained ",e)
    return aoa_arr,e_arr, L_arr

x = Random_c (10)
Q = Random_Q(10)
c= Random_c (10)

def plot_GradDesc():
    X, X_E, xf = gradient_descent (x, Q, c)
    plt.plot (X)
    plt.title ("Gradient Descent -ANGLE")
    plt. show()
    plt.vscale ("log")
    plt.plot (X_E)
    plt.title ("Gradient Descent-Error")
    plt.show ()
    plt.plot (xf)
    plt.title ("Gradient Descent -Function convergence")
    plt.show ()


def plot_OptimizedGradDescnt():
    Z, Z_E, zf = optimized_gradDesc(x, Q, c)
    plt.plot (Z)
    plt.title ("Optimized Gradient Descent -ANGLE")
    plt.show()
    plt. vscale ("log")
    plt.plot (Z_E)
    plt.title("OptimizedGradient Descent-Error")
    plt.show()
    plt.plot (zf)
    plt.title("Optimized Gradient Descent -Function")
    plt.show ()

def plot_GradientDescM() :
    Y, Y_E, yf = momentum(x, Q, c)
    plt.plot (Y)
    plt.title("Gradient Descent (Momentum) -ANGLE")
    plt.show()
    plt.vscale("log")
    plt.plot (Y_E)
    plt.title("Gradient Descent (Momentum)-Error")
    plt.show()
    plt.plot (yf)
    plt.title("Gradient Decent with Momentum-Function")
    plt.show()

def plot_OptimizedGradDescM() :
    L, L_E, lf = optimizedMomentum (x, Q, c)
    plt.plot (L)
    plt.title ("OPTIMIZED MOMENTUM GRADIENT DESCENT-Angle")
    plt.show ()
    plt. yscale ("log")
    plt.plot(L_E)
    plt.title("OPTIMIZED MOMENTUM GRADIENT DESCENT-Errop")

    plt.show()

    plt.plot (lf)
    plt.title("OPTIMIZED GRADIENT DESCENT WITHMOMENTUM-Function")
    plt.show()

def plot_Orthogonal () :
    K, K_E, kf = orthogonalMomentum(x, Q, c)
    plt.plot (K)
    plt.title("OPTIMIZED ORTHOGONAL MOMENTUM GRADIENT DESCENT-Angle")
    plt.show()
    plt. yscale ("log")
    plt.plot (K_E)
    plt.title ("OPTIMIZED ORTHOGONAL MOMENTUM GRADIENT DESCENT-Error")
    plt. show()
    plt.plot(kf)
    plt.title ("ORTHOGONAL OPTIMIZED GRADIENT DESCENT WITH MOMENTUM-Function")
    plt. show()