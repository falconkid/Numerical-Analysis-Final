import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


L = (3**0.5)/2.  # total length of the route

M = 100 # number of time intervals
N =100
T = float(np.pi*2./3.)  # total time elapse
dt = T/M # time interval

# Physical constants
m=1. # mass
k = 1. # gravitational accleration

# Main function to solve the track
def solve_lagrangian(start_pos,end_pos):
    # All the position points(including head and tail)
        #linearly distributed initialization
    track = np.linspace(start_pos,end_pos,M)
    
    # Track points(head and tail,excluded)
    track = np.delete(track,0,0)
    
    def total_s(track):# Total action
        # All the position points(including head and tail)
        pos = np.append(track,end_pos)
        pos = np.insert(pos,0,start_pos)
        
        def delta_s (init_pos,final_pos):#action within a time interval

            def Lagrangian(m,v,pos):
                K = 0.5* m *(v**2)
                V = 0.5 * k *(pos**2)
                return K - V

            pos_ave =( init_pos + final_pos)/2.
            v = (final_pos - init_pos)/dt
            return dt * Lagrangian(m,v,pos_ave)
        
        return np.sum(delta_s(pos[0:-1],pos[1:]))    
    
    result = opt.minimize(total_s,track)
    track = np.array(result.x)
    track = np.append(track,end_pos)
    track = np.insert(track,0,start_pos)
    return track

def force(x,v):
    return -k*x
def solve_euler_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       xr = x 
       vr = v

       x += vr*dt
       a = force(xr,vr)/m
       v += a*dt
       track.append(x)
    
   return np.array(track)

def solve_rk2_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       xr = x 
       vr = v

       xh = xr+vr*dt/2.
       ah = force(xr,vr)/m
       vh = vr+ah*dt/2.

       x += vh*dt
       a = force(xh,vh)/m
       v += a*dt
       track.append(x)
    
   return np.array(track)
    

if __name__ == '__main__':
    start = L *0
    end   = L *1.0
    v0 = 1.
    t  = np.linspace(0,T,M+1)
    tn  = np.linspace(0,T,N+1)

    lagrange = solve_lagrangian(start,end)
    euler_newton = solve_euler_newton(start,v0)
    rk2_newton = solve_rk2_newton(start,v0)
    stnd = np.sin(t)
    
    plt.plot(t, stnd, color ='blue', linewidth =2, linestyle ='-',label = 'standard')
    plt.plot(t, lagrange, color ='red', linewidth =2, linestyle ='--',label = 'lagrange')
    plt.plot(tn, euler_newton, color ='green', linewidth =2, linestyle =':',label = 'euler_newton')
    plt.plot(tn, rk2_newton, color ='yellow', linewidth =2, linestyle =':',label = 'rk2_newton')
    plt.legend(loc='upper left')
    plt.show()
