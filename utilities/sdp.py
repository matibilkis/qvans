import cvxpy as cp
from utilities.misc import ket_bra
import numpy as np


def give_choi(eta, bas):
    """
    Choi matrix for Amplitude damping channel (damping gamma)
    """
    gamma = np.sin(eta)**2 #this is because we take angles..

    K0 = np.array([[1,0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0,0]])
    krauss = [K0, K1]

    choi = np.zeros((4,4))
    for kk in range(len(bas)):
        state = bas[kk]
        action_state = np.zeros(state.shape)
        for k in krauss:
            action_state += np.dot(np.dot(k, state), k.T)
        prod = np.kron(action_state, state)
        choi += prod
    return choi

def psuc_ea(chois):
    F_0, F_1  = cp.Variable((4,4)), cp.Variable((4,4))
    sig = cp.Variable((2,2), hermitian=True)
    constraints = [sig>>0, cp.trace(sig)==1, F_0 >>0, F_1 >> 0,  (F_0 + F_1) == cp.kron(np.eye(2),sig)]
    objective = cp.Minimize(1- (0.5*(cp.trace( chois[0] @ F_0) + cp.trace(chois[1] @ F_1))))
    prob = cp.Problem(objective, constraints)
    cost = prob.solve(solver="MOSEK")
    return cost

def sdp_channel_disc(etas):
    #print("computing SDP...")
    b2 = [np.expand_dims(np.eye(2)[k],axis=0) for k in range(2)]
    basis = [ket_bra(b2[i], b2[j]) for i in range(2) for j in range(2)]
    chois = [give_choi(g, basis) for g in etas]
    return psuc_ea(chois) ##error prob actually
