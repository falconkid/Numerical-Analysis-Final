import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


L = 508.  # height of Taipei 101

N = 100 # number of time intervals
T = 10.  # total time elapse
dt = T/N # time interval

# Physical constants
m = 1. # mass
k = 1. #elastic constant
b = 1. #damping factor
g = 10.# gravitational accleration

c = 299792458  #light speed
c2 = c/1e06
c3 = c/1e07
c4 = c/1e08

def force(x,v):
    return -m*g
def move_newton(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force(x1,v1)/m*deltat
    return np.array([xn,vn])

def move_relativity(x,v,x1,v1,deltat,c):
    xn = x + v1*deltat
    gamma = (1-(v1/c)**2)**(-0.5)
    vn = v + force(x1,v1)/(m*gamma**3)*deltat
    return np.array([xn,vn])

def solve_rk4_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       x1,v1 = move_newton(x,v,x,v,dt*0.5)
       x2,v2 = move_newton(x,v,x1,v1,dt*0.5)
       x3,v3 = move_newton(x,v,x2,v2,dt)
       f = (move_newton(x,v,x,v,dt) + 2*move_newton(x,v,x1,v1,dt) + 2*move_newton(x,v,x2,v2,dt) +move_newton(x,v,x3,v3,dt) )/6.
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

def solve_rk4_relativity(start_pos,start_vel,c):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       x1,v1 = move_relativity(x,v,x,v,dt*0.5,c)
       x2,v2 = move_relativity(x,v,x1,v1,dt*0.5,c)
       x3,v3 = move_relativity(x,v,x2,v2,dt,c)
       f = (move_relativity(x,v,x,v,dt,c) + 2*move_relativity(x,v,x1,v1,dt,c) + 2*move_relativity(x,v,x2,v2,dt,c) +move_relativity(x,v,x3,v3,dt,c) )/6.
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

if __name__ == '__main__':
    start = L
    v0 = 0.
    t  = np.linspace(0,T,N+1)

    
    rk4_newton = solve_rk4_newton(start,v0)
    rk4_relativity = solve_rk4_relativity(start,v0,c)
    rk4_c2 = solve_rk4_relativity(start,v0,c2)
    rk4_c3 = solve_rk4_relativity(start,v0,c3)
    rk4_c4 = solve_rk4_relativity(start,v0,c4)
    stnd = L - g*(t**2)/2.

    print("newton deviation:"+str(np.sum((rk4_newton-stnd)**2)/len(stnd)))
    print("relativity c = 3e08 deviation:"+str(np.sum((rk4_relativity-stnd)**2)/len(stnd)))
    print("relativity c = 3e02 deviation:"+str(np.sum((rk4_c2-stnd)**2)/len(stnd)))
    print("relativity c = 3e01 deviation:"+str(np.sum((rk4_c3-stnd)**2)/len(stnd)))
    print("relativity c = 3e00 deviation:"+str(np.sum((rk4_c4-stnd)**2)/len(stnd)))

    plt.plot(t, stnd, color ='blue', linewidth =2, linestyle ='-',label = 'standard')

    plt.plot(t, rk4_newton, color ='red', linewidth =2, linestyle ='--',label = 'newton')
    plt.plot(t, rk4_relativity, color ='green', linewidth =2, linestyle =':',label = 'relativity c = 3e08')
    plt.plot(t, rk4_c2, color ='purple', linewidth =2, linestyle =':',label = 'relativity c = 3e02')
    plt.plot(t, rk4_c3, color ='black', linewidth =2, linestyle =':',label = 'relativity c = 3e01')
    plt.plot(t, rk4_c4, color ='brown', linewidth =2, linestyle =':',label = 'relativity c = 3e00')
    plt.legend(loc='lower left')
    plt.show()
