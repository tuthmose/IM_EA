## various utils to create quaternions and rotate molecules

from math import cos, sin, sqrt
from numpy.linalg import norm
import numpy as np
import cython

def rot1(Q):
    """
    rotation matrix of rapaport pag 202
    """
    R = np.zeros((3,3))
    
    R[0,0] = Q[0]**2.  + Q[1]**2 - 0.5
    R[0,1] = Q[1]*Q[2] + Q[0]*Q[3]
    R[0,2] = Q[1]*Q[3] - Q[0]*Q[2]
    
    R[1,0] = Q[1]*Q[2] - Q[0]*Q[3]
    R[1,1] = Q[0]**2.  + Q[2]**2 -0.5
    R[1,2] = Q[2]*Q[3] + Q[0]*Q[1]
    
    R[2,0] = Q[0]*Q[2] + Q[1]*Q[3]
    R[2,1] = Q[2]*Q[3] - Q[0]*Q[1]
    R[2,2] = Q[0]**2.  + Q[3]**2 - 0.5
    R = 2.*R
    return R

def rot2(Q):
    """
    rotation matrix of karney, see
    1.Karney, C. F. Quaternions in molecular modeling. 
    Journal of Molecular Graphics and Modelling 25, 595–604 (2007).
    """
    R = np.zeros((3,3))
    
    R[0,0] = Q[0]**2.  + Q[1]**2 - 0.5
    R[0,1] = Q[1]*Q[2] + Q[0]*Q[3]
    R[0,2] = Q[1]*Q[3] - Q[0]*Q[2]
    
    R[1,0] = Q[1]*Q[2] - Q[0]*Q[3]
    R[1,1] = Q[0]**2.  +  Q[2]**2 - 0.5
    R[1,2] = Q[2]*Q[3] + Q[0]*Q[1]
    
    R[2,0] = Q[0]*Q[2] + Q[1]*Q[3]
    R[2,1] = Q[2]*Q[3] - Q[0]*Q[1]
    R[2,2] = Q[0]**2. +  Q[3]**2 - 0.5
    R = 2.*R
    R = R.T
    return R
    
def rot3(Q):
    """
    rotation matrix of Evans, Omelyan, Sonneschein and Marco (mol2lab)
    """
    R = np.zeros((3,3))
    qq0 = Q[0]**2
    qq1 = Q[1]**2
    qq2 = Q[2]**2
    qq3 = Q[3]**2
    
    R[0,0] = -qq0+qq1-qq2+qq3
    R[0,1] = -2.*(Q[0]*Q[1]+Q[2]*Q[3])
    R[0,2] =  2.*(Q[1]*Q[2]-Q[0]*Q[3])
    
    R[1,0] =  2.*(Q[2]*Q[3]-Q[0]*Q[1])
    R[1,1] =  qq0-qq1-qq2+qq3
    R[1,2] = -2.*(Q[0]*Q[2]+Q[1]*Q[3])
    
    R[2,0] = 2.*(Q[1]*Q[2]+Q[0]*Q[3])
    R[2,1] = 2.*(Q[1]*Q[3]-Q[0]*Q[2])
    R[2,2] = -qq0-qq1+qq2+qq3
    return R

def rot4(Q):
    """
    rotation matrix of Evans, Omelyan, Sonneschein and Marco (lab2mol)
    """
    R = rot3(Q)
    return R.T
       
def rotateQ(u,Q,mat=1):
    """
    rotate vector by quaternion using rot. matrix
    non-numpy version
    """
    R = qu2rot(Q,mat=mat)
        
    v = np.zeros(3)
    v[0] = u[0]*R[0,0]+u[1]*R[1,0]+u[2]*R[2,0]
    v[1] = u[0]*R[0,1]+u[1]*R[1,1]+u[2]*R[2,1]
    v[2] = u[0]*R[2,0]+u[1]*R[2,1]+u[2]*R[2,2]
    return v

def qu2rot(Q,mat=1):
    """
    return rotation matrix correspoding to given quaternion
    """
    if mat == 1:
        R = rot1(Q)
    elif mat == 2:
        R = rot2(Q)
    elif mat == 3:
        R = rot3(Q)
    elif mat == 4:
        R = rot4(Q)
    return R
    
def rotateMQ(A,Q,mat=1):
    """
    rotate set of coordinates A(nx3) by
    rotation matrix R
    (numpy version)
    """
    R = qu2rot(Q,mat=mat) 
    return A @ R
    
def rot2qu(Rot,verbose=False):
    """
    convert  3x3 rotation matrix to quaternion, checking for trace
    """
    Q  = np.zeros(4)
    Tr = np.trace(Rot)
    if Tr >= 0.0:
        Q[0] = 0.5 * sqrt(Tr+1)
        Q[1] = (Rot[2,1] - Rot[1,2]) / (4.*Q[0])
        Q[2] = (Rot[0,2] - Rot[2,0]) / (4.*Q[0])
        Q[3] = (Rot[1,0] - Rot[0,1]) / (4.*Q[0])
    else:
        tmp1 = 0.5 * sqrt(1. + Rot[0,0] - Rot[1,1] - Rot[2,2])
        tmp2 = 0.5 * sqrt(1. + Rot[1,1] - Rot[0,0] - Rot[2,2])
        tmp3 = 0.5 * sqrt(1. + Rot[2,2] - Rot[0,0] - Rot[1,1])
        mag  = np.array((tmp1,tmp2,tmp3))
        imag = np.argmax(mag)
        if verbose:
            print("Trace, Component magnitudes, comp. max ",Tr,tmp1,tmp2,tmp3,imag)
        if imag == 0:
            Q[1] = tmp1
            Q[0] = (Rot[2,1] - Rot[1,2]) / (4.*Q[1]) 
            Q[2] = (Rot[0,1] + Rot[1,0]) / (4.*Q[1])
            Q[3] = (Rot[0,2] + Rot[2,0]) / (4.*Q[1])
        elif imag == 1:
            Q[2] = tmp2
            Q[0] = (Rot[0,2] - Rot[2,0]) / (4.*Q[2])
            Q[1] = (Rot[1,0] + Rot[0,1]) / (4.*Q[2])
            Q[3] = (Rot[1,2] + Rot[2,1]) / (4.*Q[2])
        elif imag == 2:
            Q[3] = tmp3
            Q[0] = (Rot[1,0] - Rot[0,1]) / (4.*Q[3])
            Q[1] = (Rot[0,2] + Rot[2,0]) / (4.*Q[3])
            Q[2] = (Rot[1,2] + Rot[2,1]) / (4.*Q[3])
    return Q

def rot2qu2(Rot):
    """
    convert  3x3 rotation matrix to quaternion, checking for trace
    non numpy version
    """
    Q  = np.zeros(4)
    Tr = np.trace(Rot)
    if Tr >= 0.0:
        Q[0] = 0.5 * sqrt(Tr+1)
        Q[1] = (Rot[2,1] - Rot[1,2]) / (4.*Q[0])
        Q[2] = (Rot[0,2] - Rot[2,0]) / (4.*Q[0])
        Q[3] = (Rot[1,0] - Rot[0,1]) / (4.*Q[0])
    else:
        q1 = 0.5 * sqrt(1. + Rot[0,0] - Rot[1,1] - Rot[2,2])
        q2 = 0.5 * sqrt(1. + Rot[1,1] - Rot[0,0] - Rot[2,2])
        q3 = 0.5 * sqrt(1. + Rot[2,2] - Rot[0,0] - Rot[1,1])
        if q1 > q2 and q1 > q3:
            Q[1] = q1
            Q[0] = (Rot[2,1] - Rot[1,2]) / (4.*Q[1]) 
            Q[2] = (Rot[0,1] + Rot[1,0]) / (4.*Q[1])
            Q[3] = (Rot[0,2] + Rot[2,0]) / (4.*Q[1])
        elif q2 > q1 and q2 > q3:
            Q[2] = q2
            Q[0] = (Rot[0,2] - Rot[2,0]) / (4.*Q[2])
            Q[1] = (Rot[1,0] + Rot[0,1]) / (4.*Q[2])
            Q[3] = (Rot[1,2] + Rot[2,1]) / (4.*Q[2])
        else:
            Q[3] = q3
            Q[0] = (Rot[1,0] - Rot[0,1]) / (4.*Q[3])
            Q[1] = (Rot[0,2] + Rot[2,0]) / (4.*Q[3])
            Q[2] = (Rot[1,2] + Rot[2,1]) / (4.*Q[3])
    return Q
    
def axis_angle(theta, vec, unit_q=True):
    """
    given angle and axis of rotation
    return quaternion
    """
    Q = np.zeros(4)
    Q[0]  = np.cos(theta/2.)
    Q[1:] = np.sin(theta/2.)*vec
    if unit_q:
        n = norm(Q)
        Q = Q/n
    return Q

def create_quat(u,v,verb=False,Evans=False):
    """
    return unit quaterion Q(scal,vec)
    from u to v
    """
    Q     = np.zeros(4)
    dot   = np.dot(u,v)
    cross = np.cross(u,v)
    lu    = norm(u)
    lv    = norm(v)
    if verb:
        print("dot,lu2^2, lv^2",dot,lu**2,lv**2)
        print("cross ",cross)
    Q[0] = dot + np.sqrt((lu**2)*(lv**2))
    if Evans:
        Q[1] = -cross[0]
        Q[2] =  cross[1]
        Q[3] = -cross[2]    
    else:
        Q[1] = cross[0]
        Q[2] = cross[1]
        Q[3] = cross[2]
    NQ = norm(Q)
    if verb: 
        print("norm",NQ)
    Q = Q/NQ
    if verb:
        print(norm(Q))
    return Q  
             
def random_vers():
    """
    return random versor in unit sphere
    """
    vec = np.ones(3)
    s = 2.
    while s > 1.:
        x = 2. * np.random.random() - 1.
        y = 2. * np.random.random() - 1.
        s = x*x + y*y 
    vec[2] = 1. - 2. * s
    s = 2. * sqrt(1. - s)
    vec[0] = s*x
    vec[1] = s*y             
    return vec

def quat_prod(P,Q):
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
        

def quat_cng(Q):
    """
    return the complex conjugate of Q
    """
    return np.array((Q[0],-Q[1],-Q[2],-Q[3]))

def quat_action(Q,v):
    """
    let Q act on vector v as QvQ*
    """
    Q_star = quat_cng(Q)
    v1 = quat_prod(Q,v)
    v2 = quat_prod(v1,Q_star)
    return v2[1:]

def quat_action_mat(Q,M):
    """
    apply quaternion rotation to
    all rows of an input matrix
    and return rotated matrix
    """
    N = np.empty(M.shape)
    for i,v in enumerate(M):
        N[i] = quat_action(Q,v)
    return N

def genAk(a,b):
    """
    See LSQ fit for quaternions, Karney pag. 3
    """
    A = np.zeros((4,4))
    A[0] = np.array((0.,  -b[0],-b[1],-b[2]))
    A[1] = np.array((b[0], 0,  -a[2], a[1]))
    A[2] = np.array((b[1], a[2], 0, -a[0]))
    A[3] = np.array((b[2],-a[1], a[0], 0))
    return A

def akTA(a,b):
    A = genAk(a,b)
    #print(A)
    #print(A.T)
    return np.dot(A.T,A)

def FitQ(C1,C2,mask=False,weights=False,debug=False,refl=False):
    """
    See LSQ fit for quaternions, Karney pag. 3
    """
    #kwds
    #initialization
    natoms = C1.shape[0]
    assert natoms == C2.shape[0]    
    if np.all(weights) == False:
        weights = np.ones(natoms)
    if np.all(mask) == False:
        mask = np.ones(natoms)       
    W = weights.sum()
    # subtract average coordinates
    if debug:
        print("COM1\n",np.mean(C1,axis=0),\
              "\n COM2\n", np.mean(C2,axis=0),"\n")
    xprime = C1 - np.average(C1,axis=0,weights=weights)
    yprime = C2 - np.average(C2,axis=0,weights=weights)
    if refl is True:
        xprime = -xprime
    #build B
    B = np.zeros((4,4))
    for k in range(natoms):
        Ak = akTA(yprime[k]+xprime[k],yprime[k]-xprime[k])
        B  = B + mask[k]*weights[k]*Ak
    B = B/W
    if debug: 
        print("B\n",B,"\n")
        print("B - B.T\n",B - B.T,"\n")
    #eigenvalues of B
    evalues, evectors =  np.linalg.eig(B)
    order = np.argsort(evalues)
    if debug:
        print("eigenvalues\n",evalues,"\n")
        print("eigenvectors\n",evectors.T)
        print("norms")
        print(np.linalg.norm(evectors[0]),np.linalg.norm(evectors[1]),\
              np.linalg.norm(evectors[2]),np.linalg.norm(evectors[3]))
        print("\n")
    #else:
    #    print(evalues)
    evectors = evectors.T
    if refl:
        return evectors[-1]
    else:
        return evectors[0]

def rot3d_xyz(axis,theta):
    """
    return rotation matrix along x, y or z
    for given angle in degrees
    """
    theta = theta * np.pi/180.
    if axis == "x":
        R = np.array(( (1., 0., 0.,),\
                   (0., cos(theta), -sin(theta)),\
                   (0., sin(theta), cos(theta))\
                   ))
    elif axis == "y":
        R = np.array(( (cos(theta), 0., sin(theta)),\
                   (0., 1., 0),\
                   (-sin(theta), 0., cos(theta))\
                   ))
    elif axis == "z":
        R = np.array(( (cos(theta), -sin(theta), 0.),\
                   (sin(theta), cos(theta), 0.),\
                   (0., 0., 1.)\
                   ))
    else:
        raise ValueError("Axis must be either x, y or z")
    return R

def svd_rotation(C1,C2):
    """
    return rotation matrix as obtained by
    SVD of the coordinate covariance matrix
    (Kabsch algorithm)
     #see https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
    """
    Cov = np.dot(C1.T, C2)
    V, S, W = np.linalg.svd(Cov)
    d = np.sign(np.linalg.det(Cov))
    D = np.eye(3)
    D[2,2] = d
    R = np.dot(V,np.dot(D,W))
    return R

def in_tensor(M,X):
    """
    return inertia tensor given masses and coordinates
    """
    N = X.shape[0]
    TI = np.zeros((3,3))
    for i in range(N):
        TI[0,0] = TI[0,0] + M[i]*(X[i,1]*X[i,1]+X[i,2]*X[i,2])
        TI[0,1] = TI[0,1] + M[i]*X[i,0]*X[i,1]
        TI[0,2] = TI[0,2] + M[i]*X[i,0]*X[i,2]
        TI[1,1] = TI[1,1] + M[i]*(X[i,0]*X[i,0]+X[i,2]*X[i,2])
        TI[1,2] = TI[1,2] + M[i]*X[i,1]*X[i,2]
        TI[2,2] = TI[2,2] + M[i]*(X[i,0]*X[i,0]+X[i,1]*X[i,1])
    TI[0,1] = -TI[0,1]
    TI[0,2] = -TI[0,2]
    TI[1,2] = -TI[1,2]
    TI[1,0] =  TI[0,1]
    TI[2,0] =  TI[0,2]
    TI[2,1] =  TI[1,2]
    return TI

def rmsd(X1, X2, mask=None):
    """
    calculate RMSD matrix between
    X1 and X2 assuming that rototranslation
    has been done
    """
    if mask is None:
        mask = list(range(X1.shape[0]))
    assert X1[mask].shape == X2[mask].shape
    rmsd = np.linalg.norm(X1-X2,axis=1)**2
    return np.sqrt(np.sum(rmsd)/X1.shape[0])