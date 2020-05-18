
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


L = 1.  # total length of the route

M = 50 # number of time intervals

T = float(np.pi/2)  # total time elapse


dt = T/M # time interval

# Physical constants
m=1. # mass
k = 1. # gravitational accleration


def delta_s (init_pos,final_pos):#action within a time interval

    def Lagrangian(m,v,pos):
        K = 0.5* m *(v**2)
        V = 0.5 * k *(pos**2)
        return K - V

    pos_ave =( init_pos + final_pos)/2.
    v = (final_pos - init_pos)/dt
    return dt * Lagrangian(m,v,pos_ave)

def solve_lagrangian(start_pos,v0):
    # All the position points(including head and tail)
        #linearly distributed initialization
    track = np.linspace(start_pos,v0*T,M)
    
    # Track points(head and tail,excluded)
    track = np.delete(track,0)
    
    

    def deviation(end_pos):
        def total_s(track):
            pos = np.append(track,end_pos)
            pos = np.insert(pos,0,start_pos)
            return np.sum(delta_s(pos[0:-1],pos[1:]))
        r = opt.minimize(total_s,track).x
        return ((r[0]-start_pos)/dt-v0)**2

    def best_track(end_pos):
        def total_s(track):
            pos = np.append(track,end_pos)
            pos = np.insert(pos,0,start_pos)
            return np.sum(delta_s(pos[0:-1],pos[1:]))
        r = opt.minimize(total_s,track).x
        return r

    
    result = opt.minimize(deviation,0.)
    print(result.x)
    track = best_track(result.x)
    track = np.append(track,result.x)
    track = np.insert(track,0,start_pos)
    return track



if __name__ == '__main__':
    start = L *0
    end   = L *0
    v0 = -L
    t  = np.linspace(0,T,M+1)
    

    lagrange = solve_lagrangian(start,v0)
       
    stnd = np.sin(-t)
    
    plt.plot(t, stnd, color ='blue', linewidth =2, linestyle ='-',label = 'standard')
    plt.plot(t, lagrange, color ='red', linewidth =2, linestyle ='--',label = 'lagrange')
   
    
    plt.legend(loc='upper left')
    plt.show()

