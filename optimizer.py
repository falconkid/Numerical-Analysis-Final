import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


L = 5.

M = 100
T = 1.

dt = T/M

g = 10.
def solve_laplacian(start_pos,end_pos):
    
    positions = np.linspace(start_pos,end_pos,M)
    ###################
    track = np.delete(positions,0,0)
    positions = np.append(positions,end_pos)
    m=1.
    def total_s(track):
        pos = np.append(track,end_pos)
        pos = np.insert(pos,0,start_pos)
        def delta_s (init_pos,final_pos):

            def Lagrangian(m,v,pos):
                K = 0.5* m *(v**2)
                V = -m * g * pos
                return K - V

            pos_ave =( init_pos + final_pos)/2.
            v = (final_pos - init_pos)/dt
            return dt * Lagrangian(m,v,pos_ave)
        
        return np.sum(delta_s(pos[0:-1],pos[1:]))    
    ###################
    
    result = opt.minimize(total_s,track)
    track = np.array(result.x)
    track = np.append(track,end_pos)
    track = np.insert(track,0,start_pos)
    return track

if __name__ == '__main__':
    start = L *0
    end   = L *1.0
    t  = np.linspace(0,T,M+1)
    track = solve_laplacian(start,end)
    stnd = 5. *(t**2)
    
    
    plt.plot(t, track, color ='blue', linewidth =2, linestyle ='-')
    plt.plot(t, stnd, color ='red', linewidth =2, linestyle ='--')
    plt.show()
    