import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp

N = 10000
L = 50.

M = 100
T = 1.

dl = L/N
dt = T/M

g = 10.
def viterbi(start_pos,end_pos):
    
    ###################
    start_id = int(start_pos/dl)
    action = np.zeros((N+1,M-1),dtype = float)
    index = np.zeros((N+1,M-1),dtype = int)
    positions = np.zeros((N+1),dtype = float)
    m = 1.

    def position(id):
        return id*dl
    for i in range(N+1):
        positions[i]= position(i)

    def Lagrangian(m,v,pos):
        K = 0.5* m *(v**2)
        V = m * g * pos
        return K - V
    
    """def delta_s (init_id,end_id):
        pos_ave =( position(init_id) + position(end_id))/2.
        v = (position(end_id) - position(init_id))/dt
        return dt * Lagrangian(m,v,pos_ave)"""
    def delta_s (init_pos,final_pos):
        pos_ave =( init_pos + final_pos)/2.
        v = (final_pos - init_pos)/dt
        return dt * Lagrangian(m,v,pos_ave)
    
    start_pos_pseudo = np.zeros(N+1)
    start_pos_pseudo[:] = start_pos
    action[:,0] = delta_s(start_pos_pseudo,positions)
    index [:,0] = start_id
    
    for t in range(1,M-1):
        if t%50==0:
            print("term")
        for i in range(N+1):
            pseudo_pos = np.zeros(N+1)
            pseudo_pos[:] = position(i)
            action[i,t] = np.min( action[:,t-1] + delta_s(positions,pseudo_pos) )
            index [i,t] = np.argmin( action[:,t-1] + delta_s(positions,pseudo_pos) )
    
    print(index)
    print(action)

    end_pos_pseudo = np.zeros(N+1)
    end_pos_pseudo[:] = end_pos
    action_final =  np.min(action[:,M-3]+delta_s(positions,end_pos_pseudo)) #delta_s(start_pos_pseudo,positions)
    index_final = np.argmin(action[:,M-3]+delta_s(positions,end_pos_pseudo))
    
    track = []
    track.append(index_final)
    id = index_final
    for t in range(0,M-2):
        track.insert(0,index[id][M-2-t])
        id  = index[id][M-2-t]
    
    ###################
    return track

if __name__ == '__main__':
    start = L *0.6
    end   = L *0.9
    track = viterbi(start,end)
    print(track)
    