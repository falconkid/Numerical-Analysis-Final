import numpy as np
import matplotlib.pyplot as plt

# Physical constants
m = 1. #mercury mass
M = 1.989e30 #sun mass

G = 6.67e-11# gravitational constant

c = 299792458  #light speed#You can touch

L_peri = 46.001200e09  
V_peri = 58976.
L = m*V_peri*L_peri

E = 0.5*(V_peri**2) -G*M/L_peri
a = -G*M/(2*E)

alpha = 3*(L**2)/(c**2)/(m**2)*5e05#Adjustment Parameter#You can touch

P = 10 #Number of rotation#You can touch

N = 1000*P # number of time intervals
T = 2*np.pi*(a**1.5)/((G*M)**0.5)*P # total time elapse

dt = T/N # time interval
Period = 2*np.pi*(a**1.5)/((G*M)**0.5)#period of one circle

def force_add(x,v):
    fc = -G*M*m*x/((np.sum(x**2))**1.5)
    fr = -G*M*m*x/((np.sum(x**2))**2.5)*alpha
    return fc+fr

def move_newton(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force_add(x1,v1)/m*deltat
    return np.array([xn,vn])

def solve_rk4_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]

   for t in range(N):
       x1,v1 = move_newton(x,v,x,v,dt*0.5)
       x2,v2 = move_newton(x,v,x1,v1,dt*0.5)
       x3,v3 = move_newton(x,v,x2,v2,dt)
       f = (move_newton(x,v,x,v,dt) + 2*move_newton(x,v,x1,v1,dt) + 2*move_newton(x,v,x2,v2,dt) +move_newton(x,v,x3,v3,dt) )/6.
       x,v = f[0],f[1]
       track.append(x)
   
   return np.array(track)

if __name__ == '__main__':
    start = np.array([L_peri,0])
    v0 = np.array([0,V_peri])
    t  = np.linspace(0,T,N+1)
    
    option ='newton'
    if option=='newton':

        rk4_newton = solve_rk4_newton(start,v0)
        x_newton = rk4_newton[:,0]
        y_newton = rk4_newton[:,1]

        init_angle = np.arctan((rk4_newton[0]-rk4_newton[half])[1]/(rk4_newton[0]-rk4_newton[half])[0])
        last_angle = np.arctan((rk4_newton[-1]-rk4_newton[-1-half])[1]/(rk4_newton[-1]-rk4_newton[-1-half])[0])
        print('precession angle:'+str(last_angle-init_angle))
        
        plt.plot(x_newton,y_newton, color ='green', linewidth =1, linestyle ='-',label = 'newton')
    
    plt.legend(loc='lower left')
    plt.show()
        
    
