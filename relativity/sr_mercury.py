import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Physical constants
m = 1. #mercury mass
M = 1.989e30 #sun mass

G = 6.67e-11# gravitational constant

c = 3e07#299792458  #light speed
e = 0.20563
L_peri = 46.001200e09  
V_peri = 58976.
L = m*V_peri*L_peri
E = 0.5*(V_peri**2) -G*M/L_peri
a = -G*M/(2*E)

P = 10

N = 1000*P # number of time intervals
T = 2*np.pi*(a**1.5)/((G*M)**0.5)*P # total time elapse
dt = T/N # time interval
Period = 2*np.pi*(a**1.5)/((G*M)**0.5)
half = int(Period/dt/2)





def force(x,v):
    fc = -G*M*m*x/((np.sum(x**2))**1.5)
    
    return fc
def force_add(x,v):
    fc = -G*M*m*x/((np.sum(x**2))**1.5)
    fr = -3*G*M*(L**2)*x/((np.sum(x**2))**2.5)/(c**2)/m
    return fc+fr

def move_newton(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force_add(x1,v1)/m*deltat
    return np.array([xn,vn])

def move_relativity(x,v,x1,v1,deltat,c):
    xn = x + v1*deltat

    gamma = (1-np.sum(v1**2)/(c**2))**(-0.5)
    g = force(x1,v1)/m
    v_norm = (np.sum(v1**2))
    g_p = v1*np.dot(g,v1)/v_norm
    g_o = g-g_p

    vn = v + (g_p/(gamma**3) +g_o/(gamma)) *deltat
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
        print('newton error:'+str(last_angle-init_angle))
        print('expectation:'+str( 6*np.pi*G*M/(c**2)/a/(1-e**2) *P/100))
        plt.plot(x_newton,y_newton, color ='red', linewidth =2, linestyle ='--',label = 'newton')

    
    else:
        rk4_relativity = solve_rk4_relativity(start,v0,c)
        x_relativity = rk4_relativity[:,0]
        y_relativity = rk4_relativity[:,1]

        init_angle = np.arctan((rk4_relativity[0]-rk4_relativity[half])[1]/(rk4_relativity[0]-rk4_relativity[half])[0])
        last_angle = np.arctan((rk4_relativity[-1]-rk4_relativity[-1-half])[1]/(rk4_relativity[-1]-rk4_relativity[-1-half])[0])
        print('relativity error:'+str(last_angle-init_angle))
        print('expectation:'+str( 6*np.pi*G*M/(c**2)/a/(1-e**2) *P/100))

        """relativity_error = 1/(np.sum((rk4_relativity[-1])**2))**0.5 -1/(np.sum((rk4_relativity[0])**2))**0.5
        print(relativity_error)
        print('expectation:'+str(G*M*(m**2)*e/(L**2)*(np.cos(6*P*np.pi*((G*M*m/(c*L))**2))-1)))"""

        

        plt.plot(x_relativity,y_relativity, color ='green', linewidth =2, linestyle =':',label = 'relativity')
    
    
    
    plt.legend(loc='lower left')
    plt.show()
