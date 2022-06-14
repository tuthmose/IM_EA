import numpy as np

### some utils to create new configurations
def unit_vector():
    """
    return a unit vector with a random orientation
    from origin
    """
    s = 2.
    while s > 1:
        xy = 2.*np.random.rand(2) - 1. 
        s  = xy[0]*xy[0] + xy[1]*xy[1]
        z = 1. -2.*s
    s = 2.*np.sqrt(1.-s)
    xy = s*xy
    v = np.array((xy[0],xy[1],z))
    return v

def reflect(vec,vn):
    """
    reflec vector vec
    by plane normal to vn
    """
    dp = 2.*(np.dot(vec,vn)/np.dot(vn,vn))*vn
    return vec - dp

def qnorm(Q):
    qn = np.sqrt(np.sum(Q**2))
    Q = Q/qn
    return Q

def unit_quat(angle):
    """
    create a unit quaternion with a random axis
    """
    Q = np.empty(4)
    Q[0] = np.cos(0.5*angle)
    Q[1:] = 0.5*np.sin(angle)*unit_vector()
    Q = qnorm(Q)
    return Q

def qprod(P,Q):
    """
    quaternion product in scalar,vector form
    first argument always a quaternion
    """
    if len(P) != 4:
        raise ValueError("first argument must be a quaternion")
    elif len(Q) == 4:
        T = np.zeros(4)
        T[0]  = P[0]*Q[0] - np.dot(P[1:],Q[1:])
        T[1:] = P[0]*Q[1:] + Q[0]*P[1:] + np.cross(P[1:],Q[1:])
        return T
    elif len(Q) == 3:
        T = np.zeros(4)
        T[0]  = - np.dot(P[1:],Q[0:])
        T[1:] = P[0]*Q + np.cross(P[1:],Q)
        return T
    elif len(Q) == 1:
        return Q[0]*P
        
def qstar(Q):
    """
    return the complex conjugate of Q
    """
    return np.array((Q[0],-Q[1],-Q[2],-Q[3]))

def quat_action(Q,v):
    """
    let Q act on vector v as QvQ*
    """
    Q_star = qstar(Q)
    v1 = qprod(Q,v)
    v2 = qprod(v1,Q_star)
    return v2[1:]
