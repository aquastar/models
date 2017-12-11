import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, eig
import pandas as pd
plt.style.use('ggplot')



A = np.array([
    [1,3,5],
    [3,2,2],
    [5,2,1]
])

(sign, logdet) = np.linalg.slogdet(A)

#print (sign, logdet)
#exit(0)

u, U = eig(A)

n = 2
d = 3

print 'see, the original U'
#print(U)
#print(u)
U_inv = inv(U)
U_tra = np.transpose(U)

#print 'verify u_inv and u_trans'
#print (U_inv)
#print 1/U
#print np.identity(3)/U

#print (U_tra)

#print 'recover A'
#print(np.dot(np.dot(U, np.diag(u)),U_inv))

print ''
print 'Before'
print '====='

before_numerator = np.dot(U, n*U_inv)
before_denominator = d * np.diag(u)

print np.dot(U, np.dot(n, np.dot(inv(d*np.diag(u)), U_inv)))

#print np.dot(n, np.dot(U, np.dot(inv(d*np.diag(u)), U_inv)))
#
#print np.dot(n, np.dot(inv(np.dot(d*np.diag(u), U_inv)), U_inv))
#
#print np.dot(n, inv(np.dot(U, np.dot(d, np.dot(np.diag(u), U_inv)))))
#


print ''
print 'Afters'
print np.dot(n, inv(np.dot(d, A)))




