from numpy import arange,array,ones,linalg
import pylab
from pylab import plot,show
import numpy as np
import math

#part 1
P1=np.array([])
Q1=np.array([])
file = open("censusdata.csv",'r')
for row in file:
    val=row.strip().split(',')
    P1 = np.append(P1, float(val[0]))
    Q1=np.append(Q1, math.log(float(val[1])))


A = array([ P1, ones(21)])
# linearly generated sequence

w = linalg.lstsq(A.T, Q1)[0] # obtaining the parameters
# plotting the line
line1 = w[0]*P1+w[1] # regression line
print "alpha"
print math.exp(w[1])

print "beta"
print -1*w[0]

print "actual sum squared error :"
asum=0.0
for i in range(0,21):
    asum=asum +math.pow((math.exp(line1[i])-math.exp(Q1[i])),2)

print asum

print "Value of squared error for log:"
lsum=0.0
for i in range(0,21):
    lsum=lsum +pow((line1[i]-Q1[i]),2)
print lsum


pylab.plot(P1,line1,'r-',P1,Q1,'o')
pylab.xlabel('Years')
pylab.ylabel('Population')
pylab.title('Question 5')
pylab.savefig('Answer5')
show()

#part2
P2=np.array([])
Q2=np.array([])

file = open("censusdata.csv",'r')
for row in file:
    val=row.strip().split(',')
    P2 = np.append(P2, (float(val[0])-1790))
    Q2=np.append(Q2, math.log(float(val[1])))

B = array([ P2, ones(21)])
# linearly generated sequence

w1 = linalg.lstsq(B.T,Q2)[0] # obtaining the parameters

# plotting the line
line2 = w1[0]*P2+w1[1] # regression line

print "alpha"
print math.exp(w1[1])

print "beta"
print -1*w1[0]

print "actual sum squared error :"
asum=0.0
for i in range(0,21):
    asum=asum +math.pow((math.exp(line2[i])-math.exp(Q2[i])),2)
print asum

print "Value of squared error for log:"
lsum=0.0
for i in range(0,21):
    lsum=lsum +pow((line2[i]-Q2[i]),2)
print lsum

pylab.plot(P2,line2 ,'r-',P2,Q2,'o')
pylab.xlabel('Years')
pylab.ylabel('Population (scaled)')
pylab.title('Question 6')
pylab.savefig('Answer6')
show()


#part3
from numpy.polynomial import Polynomial

p = Polynomial.fit(P1, Q1, 1)
pylab.plot(P1,Q1,'o')
pylab.plot(*p.linspace())
pylab.xlabel('Years')
pylab.ylabel('Population')
pylab.title('Question 7 (degree 1)')
pylab.savefig('Answer7one')
show()



p = Polynomial.fit(P1 ,Q1, 2)
pylab.plot(P1,Q1,'o')
pylab.plot(*p.linspace())

pylab.xlabel('Years')
pylab.ylabel('Population')
pylab.title('Question 7 (degree2)')
pylab.savefig('Answer7two')
show()


p = Polynomial.fit(P1, Q1, 3)
pylab.plot(P1,Q1,'o')
pylab.plot(*p.linspace())
pylab.xlabel('Years')
pylab.ylabel('Population')
pylab.title('Question 7 (degree3)')
pylab.savefig('Answer7three')
show()


