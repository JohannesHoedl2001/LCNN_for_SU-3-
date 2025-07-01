"""
    SU(3) group and algebra functions
"""
from lge_cnn.ym.numba_target import myjit

import os
import math
import numpy as np

"""
    SU(3) group & algebra functions
"""

N_C = 3
ALGEBRA_ELEMENTS = 8
GROUP_ELEMENTS = 9

su_precision = os.environ.get('PRECISION', 'double')

if su_precision == 'single':
    print("Using single precision")
    GROUP_TYPE = np.float32
    GROUP_TYPE_REAL = np.float32
    GROUP_TYPE_COMPLEX = np.complex64
elif su_precision == 'double':
    print("Using double precision")
    GROUP_TYPE = np.float64
    GROUP_TYPE_REAL = np.float64
    GROUP_TYPE_COMPLEX = np.complex128
else:
    print("Unsupported precision: " + su_precision)


def complex_tuple(*t):
    return tuple(map(GROUP_TYPE_COMPLEX, t))

# For conversion to matrices
zero = complex_tuple(0, 0, 0, 0, 0, 0, 0, 0, 0)
id0 = complex_tuple(1, 0, 0, 0, 1, 0, 0, 0, 1)
gm1 = complex_tuple(0, 1, 0, 1, 0, 0, 0, 0, 0)
gm2 = complex_tuple(0, -1j, 0, 1j, 0, 0, 0, 0, 0)
gm3 = complex_tuple(1, 0, 0, 0, -1, 0, 0, 0, 0)
gm4 = complex_tuple(0, 0, 1, 0, 0, 0, 1, 0, 0)
gm5 = complex_tuple(0, 0, -1j, 0, 0, 0, 1j, 0, 0)
gm6 = complex_tuple(0, 0, 0, 0, 0, 1, 0, 1, 0)
gm7 = complex_tuple(0, 0, 0, 0, 0, -1j, 0, 1j, 0)
gm8 = complex_tuple((1 / math.sqrt(3.0)), 0, 0, 0, (1 / math.sqrt(3.0)), 0, 0, 0, -2 * (1 / math.sqrt(3.0)))

# TODO: Delete *0.5, unnecessary
@myjit
def get_algebra_element(algebra_factors):
    r0 = GROUP_TYPE(0.)
    r1 = GROUP_TYPE(algebra_factors[0] * 0.5)
    r2 = GROUP_TYPE(algebra_factors[1] * 0.5)
    r3 = GROUP_TYPE(algebra_factors[2] * 0.5)
    r4 = GROUP_TYPE(algebra_factors[3] * 0.5)
    r5 = GROUP_TYPE(algebra_factors[4] * 0.5)
    r6 = GROUP_TYPE(algebra_factors[5] * 0.5)
    r7 = GROUP_TYPE(algebra_factors[6] * 0.5)
    r8 = GROUP_TYPE(algebra_factors[7] * 0.5)

    return r0, r1, r2, r3, r4, r5, r6, r7, r8


# SU(3) multiplication written in U = b0 * id0 + b1 * gm1 + ... + b8 * gm8
@myjit
def mul(a, b):
    
    r0 = (a[0] * b[0] + (2 / 3) * a[1] * b[1] + (2 / 3) * a[2] * b[2] + (2 / 3) * a[3] * b[3] + (2 / 3) * a[4] * b[4] + (2 / 3) * a[5] * b[5] + (2 / 3) * a[6] * b[6] 
          + (2 / 3) * a[7] * b[7] + (2 / 3) * a[8] * b[8])
    
    r1 = (a[0] * b[1] + a[1] * b[0] + (1 / math.sqrt(3.0)) * a[1] * b[8] + 1j * a[2] * b[3] - 1j * a[3] * b[2] + (1 / 2) * a[4] * b[6] + (1j / 2) * a[4] * b[7] - (1j / 2) * a[5] * b[6]
          + (1 / 2) * a[5] * b[7] + (1 / 2) * a[6] * b[4] + (1j / 2) * a[6] * b[5] - (1j / 2) * a[7] * b[4] + (1 / 2) * a[7] * b[5] + (1 / math.sqrt(3.0)) * a[8] * b[1])
    
    r2 = (a[0] * b[2] - 1j * a[1] * b[3] + a[2] * b[0] + (1 / math.sqrt(3.0)) * a[2] * b[8] + 1j * a[3] * b[1] + (1j / 2) * a[4] * b[6] - (1 / 2) * a[4] * b[7] + (1j / 2) * a[5] * b[7]
          + (1 / 2) * a[5] * b[6] - (1j / 2) * a[6] * b[4] + (1 / 2) * a[6] * b[5] - (1 / 2) * a[7] * b[4] - (1j / 2) * a[7] * b[5] + (1 / math.sqrt(3.0)) * a[8] * b[2])
    
    r3 =(a[0] * b[3] + 1j * a[1] * b[2] - 1j * a[2] * b[1] + a[3] * b[0] + (1 / 2) * a[4] * b[4] + (1j / 2) * a[4] * b[5] + (1 / math.sqrt(3.0)) * a[3] * b[8] - (1j / 2) * a[5] * b[4]
         + (1 / 2) * a[5] * b[5] - (1 / 2) * a[6] * b[6] - (1j / 2) * a[6] * b[7] + (1j / 2) * a[7] * b[6] - (1 / 2) * a[7] * b[7] + (1 / math.sqrt(3.0)) * a[8] * b[3])
    
    r4 =(a[0] * b[4] + (1 / 2) * a[1] * b[6] - (1j / 2) * a[1] * b[7] - (1j / 2) * a[2] * b[6] - (1 / 2) * a[2] * b[7] + (1 / 2) * a[3] * b[4] - (1j / 2) * a[3] * b[5] + a[4] * b[0] 
         - (1 / (2 * math.sqrt(3.0))) * a[4] * b[8] + (1j / 2) * a[5] * b[3] + (math.sqrt(3.0) / 2) * 1j * a[5] * b[8] + (1 / 2) * a[6] * b[1] + (1j / 2) * a[6] * b[2]
         + (1j / 2) * a[7] * b[1] - (1 / 2) * a[7] * b[2] - (1 / (2 * math.sqrt(3.0))) * a[8] * b[4] - (math.sqrt(3.0) / 2) * 1j * a[8] * b[5] + (1 / 2) * a[4] * b[3])
    
    r5 =(a[0] * b[5] + (1j / 2) * a[1] * b[6] + (1 / 2) * a[1] * b[7] + (1 / 2) * a[2] * b[6] - (1j / 2) * a[2] * b[7] + (1j / 2) * a[3] * b[4] + (1 / 2) * a[3] * b[5]
         - (1j / 2) * a[4] * b[3] - (math.sqrt(3.0) / 2) * 1j * a[4] * b[8] + a[5] * b[0] + (1 / 2) * a[5] * b[3] 
         - (1 / (2 * math.sqrt(3.0))) * a[5] * b[8] - (1j / 2) * a[6] * b[1] + (1 / 2) * a[6] * b[2] + (1 / 2) * a[7] * b[1] + (1j / 2) * a[7] * b[2] 
         + (math.sqrt(3.0) / 2) * 1j * a[8] * b[4] - (1 / (2 * math.sqrt(3.0))) * a[8] * b[5])
    
    r6 =(a[0] * b[6] + (1 / 2) * a[1] * b[4] - (1j / 2) * a[1] * b[5] + (1j / 2) * a[2] * b[4] + (1 / 2) * a[2] * b[5] - (1 / 2) * a[3] * b[6] + (1j / 2) * a[3] * b[7] + (1 / 2) * a[4] * b[1]
         - (1j / 2) * a[4] * b[2] + (1j / 2) * a[5] * b[1] + (1 / 2) * a[5] * b[2] + a[6] * b[0] - (1 / 2) * a[6] * b[3] - (1 / (2 * math.sqrt(3.0))) * a[6] * b[8] - (1j / 2) * a[7] * b[3]
         + (math.sqrt(3.0) / 2) * 1j * a[7] * b[8] - (math.sqrt(3.0) / 2) * 1j * a[8] * b[7] - (1 / (2 * math.sqrt(3.0))) * a[8] * b[6])
    
    r7 =(a[0] * b[7] + (1j / 2) * a[1] * b[4] + (1 / 2) * a[1] * b[5] - (1 / 2) * a[2] * b[4] + (1j / 2) * a[2] * b[5] - (1j / 2) * a[3] * b[6] - (1 / 2) * a[3] * b[7] - (1j / 2) * a[4] * b[1]
         - (1 / 2) * a[4] * b[2] + (1 / 2) * a[5] * b[1] - (1j / 2) * a[5] * b[2] + (1j / 2) * a[6] * b[3] - (math.sqrt(3.0) / 2) * 1j * a[6] * b[8] + a[7] * b[0] - (1 / 2) * a[7] * b[3]
         - (1 / (2 * math.sqrt(3.0))) * a[7] * b[8] + (math.sqrt(3.0) / 2) * 1j * a[8] * b[6] - (1 / (2 * math.sqrt(3.0))) * a[8] * b[7])
    
    r8 =(a[0] * b[8] + (1 / math.sqrt(3.0)) * a[1] * b[1] + (1 / math.sqrt(3.0)) * a[2] * b[2] + (1 / math.sqrt(3.0)) * a[3] * b[3] - (1 / (2 * math.sqrt(3.0))) * a[4] * b[4]
         + (math.sqrt(3.0) / 2) * 1j * a[4] * b[5] - (math.sqrt(3.0) / 2) * 1j * a[5] * b[4] - (1 / (2 * math.sqrt(3.0))) * a[5] * b[5] - (1 / (2 * math.sqrt(3.0))) * a[6] * b[6]
         + (math.sqrt(3.0) / 2) * 1j * a[6] * b[7] - (math.sqrt(3.0) / 2) * 1j * a[7] * b[6] - (1 / (2 * math.sqrt(3.0))) * a[7] * b[7] + a[8] * b[0] - (1 / math.sqrt(3.0)) * a[8] * b[8])
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

"""
# SU(3) multiplication
@myjit
def mul(a, b): # TODO: rename to matmul (as in numpy)
    am = to_matrix(a)
    bm = to_matrix(b)
    
    r0 = am[0] * bm[0] + am[1] * bm[3] + am[2] * bm[6]
    r1 = am[0] * bm[1] + am[1] * bm[4] + am[2] * bm[7]
    r2 = am[0] * bm[2] + am[1] * bm[5] + am[2] * bm[8]
    r3 = am[3] * bm[0] + am[4] * bm[3] + am[5] * bm[6]
    r4 = am[3] * bm[1] + am[4] * bm[4] + am[5] * bm[7]
    r5 = am[3] * bm[2] + am[4] * bm[5] + am[5] * bm[8]
    r6 = am[6] * bm[0] + am[7] * bm[3] + am[8] * bm[6]
    r7 = am[6] * bm[1] + am[7] * bm[4] + am[8] * bm[7]
    r8 = am[6] * bm[2] + am[7] * bm[5] + am[8] * bm[8]
                                                    
    retnum = to_repr((r0, r1, r2, r3, r4, r5, r6, r7, r8))
    
    ret0 = retnum[0]
    ret1 = retnum[1]
    ret2 = retnum[2]
    ret3 = retnum[3]
    ret4 = retnum[4]
    ret5 = retnum[5]
    ret6 = retnum[6]
    ret7 = retnum[7]
    ret8 = retnum[8]
    
    return ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8 
""" 
    
# Exponential map: v real valued!
@myjit
def mexp(v):
    """
    https://arxiv.org/abs/2207.02167
    """
    S0 = GROUP_TYPE_COMPLEX(0.0)
    S1 = GROUP_TYPE_COMPLEX(0.0)
    S2 = GROUP_TYPE_COMPLEX(0.0)
    S3 = GROUP_TYPE_COMPLEX(0.0)
    S4 = GROUP_TYPE_COMPLEX(0.0)
    S5 = GROUP_TYPE_COMPLEX(0.0)
    S6 = GROUP_TYPE_COMPLEX(0.0)
    S7 = GROUP_TYPE_COMPLEX(0.0)
    S8 = GROUP_TYPE_COMPLEX(0.0)
    eta = GROUP_TYPE_REAL(0.0)
    psi = GROUP_TYPE_REAL(0.0)
    z1, z2, z3 = GROUP_TYPE_REAL(0.0), GROUP_TYPE_REAL(0.0), GROUP_TYPE_REAL(0.0)
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = (GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                          GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                          GROUP_TYPE_COMPLEX(0.0))
    
    norm = GROUP_TYPE_REAL(math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
    if norm < 1e-6:
        return 1, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        vnorm0 = v[0] / norm # Can delete
        vnorm1 = v[1] / norm
        vnorm2 = v[2] / norm
        vnorm3 = v[3] / norm
        vnorm4 = v[4] / norm
        vnorm5 = v[5] / norm
        vnorm6 = v[6] / norm
        vnorm7 = v[7] / norm
        vnorm8 = v[8] / norm
    
        S0 = vnorm1 * gm1[0] + vnorm2 * gm2[0] + vnorm3 * gm3[0] + vnorm4 * gm4[0] + vnorm5 * gm5[0] + vnorm6 * gm6[0] + vnorm7 * gm7[0] + vnorm8 * gm8[0]
        S1 = vnorm1 * gm1[1] + vnorm2 * gm2[1] + vnorm3 * gm3[1] + vnorm4 * gm4[1] + vnorm5 * gm5[1] + vnorm6 * gm6[1] + vnorm7 * gm7[1] + vnorm8 * gm8[1]
        S2 = vnorm1 * gm1[2] + vnorm2 * gm2[2] + vnorm3 * gm3[2] + vnorm4 * gm4[2] + vnorm5 * gm5[2] + vnorm6 * gm6[2] + vnorm7 * gm7[2] + vnorm8 * gm8[2]
        S3 = vnorm1 * gm1[3] + vnorm2 * gm2[3] + vnorm3 * gm3[3] + vnorm4 * gm4[3] + vnorm5 * gm5[3] + vnorm6 * gm6[3] + vnorm7 * gm7[3] + vnorm8 * gm8[3]
        S4 = vnorm1 * gm1[4] + vnorm2 * gm2[4] + vnorm3 * gm3[4] + vnorm4 * gm4[4] + vnorm5 * gm5[4] + vnorm6 * gm6[4] + vnorm7 * gm7[4] + vnorm8 * gm8[4]
        S5 = vnorm1 * gm1[5] + vnorm2 * gm2[5] + vnorm3 * gm3[5] + vnorm4 * gm4[5] + vnorm5 * gm5[5] + vnorm6 * gm6[5] + vnorm7 * gm7[5] + vnorm8 * gm8[5]
        S6 = vnorm1 * gm1[6] + vnorm2 * gm2[6] + vnorm3 * gm3[6] + vnorm4 * gm4[6] + vnorm5 * gm5[6] + vnorm6 * gm6[6] + vnorm7 * gm7[6] + vnorm8 * gm8[6]
        S7 = vnorm1 * gm1[7] + vnorm2 * gm2[7] + vnorm3 * gm3[7] + vnorm4 * gm4[7] + vnorm5 * gm5[7] + vnorm6 * gm6[7] + vnorm7 * gm7[7] + vnorm8 * gm8[7]
        S8 = vnorm1 * gm1[8] + vnorm2 * gm2[8] + vnorm3 * gm3[8] + vnorm4 * gm4[8] + vnorm5 * gm5[8] + vnorm6 * gm6[8] + vnorm7 * gm7[8] + vnorm8 * gm8[8]
    
        etau = GROUP_TYPE_REAL((S0 * S4 * S8 + S1 * S5 * S6 + S2 * S3 * S7 - S2 * S4 * S6 - S7 * S5 * S0 - S8 * S3 * S1).real)
        eta = max(min(etau, (2 * math.sqrt(3.0)) / 9), -(2 * math.sqrt(3.0)) / 9)

        psiu = (1 / 3) * math.acos((3 / 2) * math.sqrt(3.0) * eta)
        psi = max(0, min(psiu, math.pi / 3))
        
        z1 = 2 * math.sqrt(1 / 3) * math.cos(psi)
        z2 = -math.sin(psi) - (math.cos(psi) / math.sqrt(3.0))
        z3 = math.sin(psi) - (math.cos(psi) / math.sqrt(3.0))

        argexp1 = z1 * norm
        argexp2 = z2 * norm
        argexp3 = z3 * norm
        cospart1 = math.cos(argexp1)
        sinpart1 = math.sin(argexp1)
        expon1 = cospart1 + 1j * sinpart1
        cospart2 = math.cos(argexp2)
        sinpart2 = math.sin(argexp2)
        expon2 = cospart2 + 1j * sinpart2
        cospart3 = math.cos(argexp3)
        sinpart3 = math.sin(argexp3)
        expon3 = cospart3 + 1j * sinpart3
        denominator1 = 3 * z1 * z1 - 1
        denominator2 = 3 * z2 * z2 - 1
        denominator3 = 3 * z3 * z3 - 1
        threshold = 1e-16
        epsilon = 1e-10
        if abs(denominator1) < threshold:
            denominator1 += 3 * epsilon
        if abs(denominator2) < threshold:
            denominator2 += 3 * epsilon
        if abs(denominator3) < threshold:
            denominator3 += 3 * epsilon
        
        r0 += (expon1 / denominator1) * ((z1 * z1) - 1 + (2 / 3) * vnorm1 * vnorm1 + (2 / 3) * vnorm2 * vnorm2 + (2 / 3) * vnorm3 * vnorm3 + (2 / 3) * vnorm4 * vnorm4 + (2 / 3) * vnorm5 * vnorm5
               + (2 / 3) * vnorm6 * vnorm6 + (2 / 3) * vnorm7 * vnorm7 + (2 / 3) * vnorm8 * vnorm8)
        r0 += (expon2 / denominator2) * ((z2 * z2) - 1 + (2 / 3) * vnorm1 * vnorm1 + (2 / 3) * vnorm2 * vnorm2 + (2 / 3) * vnorm3 * vnorm3 + (2 / 3) * vnorm4 * vnorm4 + (2 / 3) * vnorm5 * vnorm5
               + (2 / 3) * vnorm6 * vnorm6 + (2 / 3) * vnorm7 * vnorm7 + (2 / 3) * vnorm8 * vnorm8)
        r0 += (expon3 / denominator3) * ((z3 * z3) - 1 + (2 / 3) * vnorm1 * vnorm1 + (2 / 3) * vnorm2 * vnorm2 + (2 / 3) * vnorm3 * vnorm3 + (2 / 3) * vnorm4 * vnorm4 + (2 / 3) * vnorm5 * vnorm5
               + (2 / 3) * vnorm6 * vnorm6 + (2 / 3) * vnorm7 * vnorm7 + (2 / 3) * vnorm8 * vnorm8)
    
        r1 += (expon1 / denominator1) * (z1 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm8 + 1j * vnorm2 * vnorm3 - 1j * vnorm3 * vnorm2 + (1 / 2) * vnorm4 * vnorm6 + (1j / 2) * vnorm4 * vnorm7
               - (1j / 2) * vnorm5 * vnorm6 + (1 / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm6 * vnorm4 + (1j / 2) * vnorm6 * vnorm5 - (1j / 2) * vnorm7 * vnorm4 + (1 / 2) * vnorm7 * vnorm5 
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm1)
        r1 += (expon2 / denominator2) * (z2 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm8 + 1j * vnorm2 * vnorm3 - 1j * vnorm3 * vnorm2 + (1 / 2) * vnorm4 * vnorm6 + (1j / 2) * vnorm4 * vnorm7
               - (1j / 2) * vnorm5 * vnorm6 + (1 / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm6 * vnorm4 + (1j / 2) * vnorm6 * vnorm5 - (1j / 2) * vnorm7 * vnorm4 + (1 / 2) * vnorm7 * vnorm5 
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm1)
        r1 += (expon3 / denominator3) * (z3 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm8 + 1j * vnorm2 * vnorm3 - 1j * vnorm3 * vnorm2 + (1 / 2) * vnorm4 * vnorm6 + (1j / 2) * vnorm4 * vnorm7
               - (1j / 2) * vnorm5 * vnorm6 + (1 / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm6 * vnorm4 + (1j / 2) * vnorm6 * vnorm5 - (1j / 2) * vnorm7 * vnorm4 + (1 / 2) * vnorm7 * vnorm5 
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm1)
    
        r2 += (expon1 / denominator1) * (z1 * vnorm2 - 1j * vnorm1 * vnorm3 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm8 + 1j * vnorm3 * vnorm1 + (1j / 2) * vnorm4 * vnorm6 - (1 / 2) * vnorm4 * vnorm7
               + (1j / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm5 * vnorm6 - (1j / 2) * vnorm6 * vnorm4 + (1 / 2) * vnorm6 * vnorm5 - (1 / 2) * vnorm7 * vnorm4 - (1j / 2) * vnorm7 * vnorm5
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm2)
        r2 += (expon2 / denominator2) * (z2 * vnorm2 - 1j * vnorm1 * vnorm3 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm8 + 1j * vnorm3 * vnorm1 + (1j / 2) * vnorm4 * vnorm6 - (1 / 2) * vnorm4 * vnorm7
               + (1j / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm5 * vnorm6 - (1j / 2) * vnorm6 * vnorm4 + (1 / 2) * vnorm6 * vnorm5 - (1 / 2) * vnorm7 * vnorm4 - (1j / 2) * vnorm7 * vnorm5
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm2)
        r2 += (expon3 / denominator3) * (z3 * vnorm2 - 1j * vnorm1 * vnorm3 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm8 + 1j * vnorm3 * vnorm1 + (1j / 2) * vnorm4 * vnorm6 - (1 / 2) * vnorm4 * vnorm7
               + (1j / 2) * vnorm5 * vnorm7 + (1 / 2) * vnorm5 * vnorm6 - (1j / 2) * vnorm6 * vnorm4 + (1 / 2) * vnorm6 * vnorm5 - (1 / 2) * vnorm7 * vnorm4 - (1j / 2) * vnorm7 * vnorm5
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm2)
    
        r3 += (expon1 / denominator1) * (z1 * vnorm3 + 1j * vnorm1 * vnorm2 - 1j * vnorm2 * vnorm1 + (1 / 2) * vnorm4 * vnorm4 + (1j / 2) * vnorm4 * vnorm5 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm8
               - (1j / 2) * vnorm5 * vnorm4 + (1 / 2) * vnorm5 * vnorm5 - (1 / 2) * vnorm6 * vnorm6 - (1j / 2) * vnorm6 * vnorm7 + (1j / 2) * vnorm7 * vnorm6 - (1 / 2) * vnorm7 * vnorm7
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm3)
        r3 += (expon2 / denominator2) * (z2 * vnorm3 + 1j * vnorm1 * vnorm2 - 1j * vnorm2 * vnorm1 + (1 / 2) * vnorm4 * vnorm4 + (1j / 2) * vnorm4 * vnorm5 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm8
               - (1j / 2) * vnorm5 * vnorm4 + (1 / 2) * vnorm5 * vnorm5 - (1 / 2) * vnorm6 * vnorm6 - (1j / 2) * vnorm6 * vnorm7 + (1j / 2) * vnorm7 * vnorm6 - (1 / 2) * vnorm7 * vnorm7
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm3)
        r3 += (expon3 / denominator3) * (z3 * vnorm3 + 1j * vnorm1 * vnorm2 - 1j * vnorm2 * vnorm1 + (1 / 2) * vnorm4 * vnorm4 + (1j / 2) * vnorm4 * vnorm5 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm8
               - (1j / 2) * vnorm5 * vnorm4 + (1 / 2) * vnorm5 * vnorm5 - (1 / 2) * vnorm6 * vnorm6 - (1j / 2) * vnorm6 * vnorm7 + (1j / 2) * vnorm7 * vnorm6 - (1 / 2) * vnorm7 * vnorm7
               + (1 / math.sqrt(3.0)) * vnorm8 * vnorm3)
    
        r4 += (expon1 / denominator1) * (z1 * vnorm4 + (1 / 2) * vnorm1 * vnorm6 - (1j / 2) * vnorm1 * vnorm7 - (1j / 2) * vnorm2 * vnorm6 - (1 / 2) * vnorm2 * vnorm7 + (1 / 2) * vnorm3 * vnorm4
               - (1j / 2) * vnorm3 * vnorm5 - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm8 + (1j / 2) * vnorm5 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm8 + (1 / 2) * vnorm6 * vnorm1
               + (1j / 2) * vnorm6 * vnorm2 + (1j / 2) * vnorm7 * vnorm1 - (1 / 2) * vnorm7 * vnorm2 - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm4 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm5
               + (1 / 2) * vnorm4 * vnorm3)
        r4 += (expon2 / denominator2) * (z2 * vnorm4 + (1 / 2) * vnorm1 * vnorm6 - (1j / 2) * vnorm1 * vnorm7 - (1j / 2) * vnorm2 * vnorm6 - (1 / 2) * vnorm2 * vnorm7 + (1 / 2) * vnorm3 * vnorm4
               - (1j / 2) * vnorm3 * vnorm5 - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm8 + (1j / 2) * vnorm5 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm8 + (1 / 2) * vnorm6 * vnorm1
               + (1j / 2) * vnorm6 * vnorm2 + (1j / 2) * vnorm7 * vnorm1 - (1 / 2) * vnorm7 * vnorm2 - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm4 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm5
               + (1 / 2) * vnorm4 * vnorm3)
        r4 += (expon3 / denominator3) * (z3 * vnorm4 + (1 / 2) * vnorm1 * vnorm6 - (1j / 2) * vnorm1 * vnorm7 - (1j / 2) * vnorm2 * vnorm6 - (1 / 2) * vnorm2 * vnorm7 + (1 / 2) * vnorm3 * vnorm4
               - (1j / 2) * vnorm3 * vnorm5 - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm8 + (1j / 2) * vnorm5 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm8 + (1 / 2) * vnorm6 * vnorm1
               + (1j / 2) * vnorm6 * vnorm2 + (1j / 2) * vnorm7 * vnorm1 - (1 / 2) * vnorm7 * vnorm2 - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm4 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm5
               + (1 / 2) * vnorm4 * vnorm3)

        r5 += (expon1 / denominator1) * (z1 * vnorm5 + (1j / 2) * vnorm1 * vnorm6 + (1 / 2) * vnorm1 * vnorm7 + (1 / 2) * vnorm2 * vnorm6 - (1j / 2) * vnorm2 * vnorm7 + (1j / 2) * vnorm3 * vnorm4
               + (1 / 2) * vnorm3 * vnorm5 - (1j / 2) * vnorm4 * vnorm3 - (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm8 + (1 / 2) * vnorm5 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm8
               - (1j / 2) * vnorm6 * vnorm1 + (1 / 2) * vnorm6 * vnorm2 + (1 / 2) * vnorm7 * vnorm1 + (1j / 2) * vnorm7 * vnorm2 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm4
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm5)
        r5 += (expon2 / denominator2) * (z2 * vnorm5 + (1j / 2) * vnorm1 * vnorm6 + (1 / 2) * vnorm1 * vnorm7 + (1 / 2) * vnorm2 * vnorm6 - (1j / 2) * vnorm2 * vnorm7 + (1j / 2) * vnorm3 * vnorm4
               + (1 / 2) * vnorm3 * vnorm5 - (1j / 2) * vnorm4 * vnorm3 - (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm8 + (1 / 2) * vnorm5 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm8
               - (1j / 2) * vnorm6 * vnorm1 + (1 / 2) * vnorm6 * vnorm2 + (1 / 2) * vnorm7 * vnorm1 + (1j / 2) * vnorm7 * vnorm2 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm4
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm5)
        r5 += (expon3 / denominator3) * (z3 * vnorm5 + (1j / 2) * vnorm1 * vnorm6 + (1 / 2) * vnorm1 * vnorm7 + (1 / 2) * vnorm2 * vnorm6 - (1j / 2) * vnorm2 * vnorm7 + (1j / 2) * vnorm3 * vnorm4
               + (1 / 2) * vnorm3 * vnorm5 - (1j / 2) * vnorm4 * vnorm3 - (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm8 + (1 / 2) * vnorm5 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm8
               - (1j / 2) * vnorm6 * vnorm1 + (1 / 2) * vnorm6 * vnorm2 + (1 / 2) * vnorm7 * vnorm1 + (1j / 2) * vnorm7 * vnorm2 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm4
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm5)
    
        r6 += (expon1 / denominator1) * (z1 * vnorm6 + (1 / 2) * vnorm1 * vnorm4 - (1j / 2) * vnorm1 * vnorm5 + (1j / 2) * vnorm2 * vnorm4 + (1 / 2) * vnorm2 * vnorm5 - (1 / 2) * vnorm3 * vnorm6
               + (1j / 2) * vnorm3 * vnorm7 + (1 / 2) * vnorm4 * vnorm1 - (1j / 2) * vnorm4 * vnorm2 + (1j / 2) * vnorm5 * vnorm1 + (1 / 2) * vnorm5 * vnorm2 - (1 / 2) * vnorm6 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm8 - (1j / 2) * vnorm7 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm8 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm7
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm6)
        r6 += (expon2 / denominator2) * (z2 * vnorm6 + (1 / 2) * vnorm1 * vnorm4 - (1j / 2) * vnorm1 * vnorm5 + (1j / 2) * vnorm2 * vnorm4 + (1 / 2) * vnorm2 * vnorm5 - (1 / 2) * vnorm3 * vnorm6
               + (1j / 2) * vnorm3 * vnorm7 + (1 / 2) * vnorm4 * vnorm1 - (1j / 2) * vnorm4 * vnorm2 + (1j / 2) * vnorm5 * vnorm1 + (1 / 2) * vnorm5 * vnorm2 - (1 / 2) * vnorm6 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm8 - (1j / 2) * vnorm7 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm8 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm7
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm6)
        r6 += (expon3 / denominator3) * (z3 * vnorm6 + (1 / 2) * vnorm1 * vnorm4 - (1j / 2) * vnorm1 * vnorm5 + (1j / 2) * vnorm2 * vnorm4 + (1 / 2) * vnorm2 * vnorm5 - (1 / 2) * vnorm3 * vnorm6
               + (1j / 2) * vnorm3 * vnorm7 + (1 / 2) * vnorm4 * vnorm1 - (1j / 2) * vnorm4 * vnorm2 + (1j / 2) * vnorm5 * vnorm1 + (1 / 2) * vnorm5 * vnorm2 - (1 / 2) * vnorm6 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm8 - (1j / 2) * vnorm7 * vnorm3 + (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm8 - (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm7
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm6)
    
        r7 += (expon1 / denominator1) * (z1 * vnorm7 + (1j / 2) * vnorm1 * vnorm4 + (1 / 2) * vnorm1 * vnorm5 - (1 / 2) * vnorm2 * vnorm4 + (1j / 2) * vnorm2 * vnorm5 - (1j / 2) * vnorm3 * vnorm6
               - (1 / 2) * vnorm3 * vnorm7 - (1j / 2) * vnorm4 * vnorm1 - (1 / 2) * vnorm4 * vnorm2 + (1 / 2) * vnorm5 * vnorm1 - (1j / 2) * vnorm5 * vnorm2 + (1j / 2) * vnorm6 * vnorm3
               - (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm8 - (1 / 2) * vnorm7 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm8 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm6
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm7)
        r7 += (expon2 / denominator2) * (z2 * vnorm7 + (1j / 2) * vnorm1 * vnorm4 + (1 / 2) * vnorm1 * vnorm5 - (1 / 2) * vnorm2 * vnorm4 + (1j / 2) * vnorm2 * vnorm5 - (1j / 2) * vnorm3 * vnorm6
               - (1 / 2) * vnorm3 * vnorm7 - (1j / 2) * vnorm4 * vnorm1 - (1 / 2) * vnorm4 * vnorm2 + (1 / 2) * vnorm5 * vnorm1 - (1j / 2) * vnorm5 * vnorm2 + (1j / 2) * vnorm6 * vnorm3
               - (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm8 - (1 / 2) * vnorm7 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm8 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm6
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm7)
        r7 += (expon3 / denominator3) * (z3 * vnorm7 + (1j / 2) * vnorm1 * vnorm4 + (1 / 2) * vnorm1 * vnorm5 - (1 / 2) * vnorm2 * vnorm4 + (1j / 2) * vnorm2 * vnorm5 - (1j / 2) * vnorm3 * vnorm6
               - (1 / 2) * vnorm3 * vnorm7 - (1j / 2) * vnorm4 * vnorm1 - (1 / 2) * vnorm4 * vnorm2 + (1 / 2) * vnorm5 * vnorm1 - (1j / 2) * vnorm5 * vnorm2 + (1j / 2) * vnorm6 * vnorm3
               - (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm8 - (1 / 2) * vnorm7 * vnorm3 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm8 + (math.sqrt(3.0) / 2) * 1j * vnorm8 * vnorm6
               - (1 / (2 * math.sqrt(3.0))) * vnorm8 * vnorm7)
    
        r8 += (expon1 / denominator1) * (z1 * vnorm8 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm2 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm4 + (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm5 - (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm4 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm5
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm6 + (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm7 - (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm6 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm7
               - (1 / math.sqrt(3.0)) * vnorm8 * vnorm8)
        r8 += (expon2 / denominator2) * (z2 * vnorm8 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm2 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm4 + (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm5 - (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm4 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm5
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm6 + (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm7 - (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm6 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm7
               - (1 / math.sqrt(3.0)) * vnorm8 * vnorm8)
        r8 += (expon3 / denominator3) * (z3 * vnorm8 + (1 / math.sqrt(3.0)) * vnorm1 * vnorm1 + (1 / math.sqrt(3.0)) * vnorm2 * vnorm2 + (1 / math.sqrt(3.0)) * vnorm3 * vnorm3
               - (1 / (2 * math.sqrt(3.0))) * vnorm4 * vnorm4 + (math.sqrt(3.0) / 2) * 1j * vnorm4 * vnorm5 - (math.sqrt(3.0) / 2) * 1j * vnorm5 * vnorm4 - (1 / (2 * math.sqrt(3.0))) * vnorm5 * vnorm5
               - (1 / (2 * math.sqrt(3.0))) * vnorm6 * vnorm6 + (math.sqrt(3.0) / 2) * 1j * vnorm6 * vnorm7 - (math.sqrt(3.0) / 2) * 1j * vnorm7 * vnorm6 - (1 / (2 * math.sqrt(3.0))) * vnorm7 * vnorm7
               - (1 / math.sqrt(3.0)) * vnorm8 * vnorm8)
        
        r0img = (1 / 2) * (r0 - r0.conjugate()) / 1j
        r0real = (1 / 2) * (r0 + r0.conjugate())
        r1img = (1 / 2) * (r1 - r1.conjugate()) / 1j
        r1real = (1 / 2) * (r1 + r1.conjugate())
        r2img = (1 / 2) * (r2 - r2.conjugate()) / 1j
        r2real = (1 / 2) * (r2 + r2.conjugate())
        r3img = (1 / 2) * (r3 - r3.conjugate()) / 1j
        r3real = (1 / 2) * (r3 + r3.conjugate())
        r4img = (1 / 2) * (r4 - r4.conjugate()) / 1j
        r4real = (1 / 2) * (r4 + r4.conjugate())
        r5img = (1 / 2) * (r5 - r5.conjugate()) / 1j
        r5real = (1 / 2) * (r5 + r5.conjugate())
        r6img = (1 / 2) * (r6 - r6.conjugate()) / 1j
        r6real = (1 / 2) * (r6 + r6.conjugate())
        r7img = (1 / 2) * (r7 - r7.conjugate()) / 1j
        r7real = (1 / 2) * (r7 + r7.conjugate())
        r8img = (1 / 2) * (r8 - r8.conjugate()) / 1j
        r8real = (1 / 2) * (r8 + r8.conjugate())
        
        threshold2 = 5e-16
        
        if abs(r0img) < threshold2:
            r0img = 0.0
        if abs(r0real) < threshold2:
            r0real = 0.0
        if abs(r1img) < threshold2:
            r1img = 0.0
        if abs(r1real) < threshold2:
            r1real = 0.0
        if abs(r2img) < threshold2:
            r2img = 0.0
        if abs(r2real) < threshold2:
            r2real = 0.0
        if abs(r3img) < threshold2:
            r3img = 0.0
        if abs(r3real) < threshold2:
            r3real = 0.0
        if abs(r4img) < threshold2:
            r4img = 0.0
        if abs(r4real) < threshold2:
            r4real = 0.0
        if abs(r5img) < threshold2:
            r5img = 0.0
        if abs(r5real) < threshold2:
            r5real = 0.0
        if abs(r6img) < threshold2:
            r6img = 0.0
        if abs(r6real) < threshold2:
            r6real = 0.0
        if abs(r7img) < threshold2:
            r7img = 0.0
        if abs(r7real) < threshold2:
            r7real = 0.0
        if abs(r8img) < threshold2:
            r8img = 0.0
        if abs(r8real) < threshold2:
            r8real = 0.0
        
        ret0 = r0real + 1j * r0img
        ret1 = r1real + 1j * r1img
        ret2 = r2real + 1j * r2img
        ret3 = r3real + 1j * r3img
        ret4 = r4real + 1j * r4img
        ret5 = r5real + 1j * r5img
        ret6 = r6real + 1j * r6img
        ret7 = r7real + 1j * r7img
        ret8 = r8real + 1j * r8img
            
        return ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8

# Inverse M^-1 = M^dag
@myjit
def inv(a):
    r0 = a[0].conjugate()
    r1 = a[1].conjugate()
    r2 = a[2].conjugate()
    r3 = a[3].conjugate()
    r4 = a[4].conjugate()
    r5 = a[5].conjugate()
    r6 = a[6].conjugate()
    r7 = a[7].conjugate()
    r8 = a[8].conjugate()
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

"""
# Not used for the thesis, but can be implemented easily
@myjit
def det(a):
    fm0 = a[0] + a[3] + (1 / math.sqrt(3.0)) * a[8]
    fm1 = a[1] - 1j * a[2]
    fm2 = a[4] - 1j * a[5]
    fm3 = a[1] + 1j * a[2]
    fm4 = a[0] - a[3] + (1 / math.sqrt(3.0)) * a[8]
    fm5 = a[6] - 1j * a[7]
    fm6 = a[4] + 1j * a[5]
    fm7 = a[6] + 1j * a[7]
    fm8 = a[0] - (2 / math.sqrt(3.0)) * a[8]
    
    det = fm0 * fm4 * fm8 + fm1 * fm5 * fm6 + fm2 * fm3 * fm7 - fm6 * fm4 * fm2 - fm7 * fm5 * fm0 - fm8 * fm3 * fm1
    
    return det
"""

@myjit
def det(a):
    am = to_matrix(a)
    
    det = am[0] * am[4] * am[8] + am[1] * am[5] * am[6] + am[2] * am[3] * am[7] - am[6] * am[4] * am[2] - am[7] * am[5] * am[0] - am[8] * am[3] * am[1]
    
    return det

# Anti-hermitian and traceless
@myjit
def ah(u):
    return (GROUP_TYPE_REAL(0.0), GROUP_TYPE_REAL(u[1].imag), GROUP_TYPE_REAL(u[2].imag), GROUP_TYPE_REAL(u[3].imag), GROUP_TYPE_REAL(u[4].imag), GROUP_TYPE_REAL(u[5].imag)
           , GROUP_TYPE_REAL(u[6].imag), GROUP_TYPE_REAL(u[7].imag), GROUP_TYPE_REAL(u[8].imag))

# Anti-hermitian part
@myjit
def im(u):
    return u[0].imag, u[1].imag, u[2].imag, u[3].imag, u[4].imag, u[5].imag, u[6].imag, u[7].imag, u[8].imag

# Group add: g0 = g0 + f * g1
@myjit
def add(g0, g1):
    r0 = g0[0] + g1[0]
    r1 = g0[1] + g1[1]
    r2 = g0[2] + g1[2]
    r3 = g0[3] + g1[3]
    r4 = g0[4] + g1[4]
    r5 = g0[5] + g1[5]
    r6 = g0[6] + g1[6]
    r7 = g0[7] + g1[7]
    r8 = g0[8] + g1[8]
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# Multiply by scalar
@myjit
def mul_s(g0, f):  # TODO: rename to mul
    # Unfortunately, tuple creation from list comprehension does not work in numba:
    # see https://github.com/numba/numba/issues/2771
    #
    # result = tuple(f * g0[i] for i in range(4))
    # return result
    r0 = GROUP_TYPE_COMPLEX(f) * g0[0]
    r1 = GROUP_TYPE_COMPLEX(f) * g0[1]
    r2 = GROUP_TYPE_COMPLEX(f) * g0[2]
    r3 = GROUP_TYPE_COMPLEX(f) * g0[3]
    r4 = GROUP_TYPE_COMPLEX(f) * g0[4]
    r5 = GROUP_TYPE_COMPLEX(f) * g0[5]
    r6 = GROUP_TYPE_COMPLEX(f) * g0[6]
    r7 = GROUP_TYPE_COMPLEX(f) * g0[7]
    r8 = GROUP_TYPE_COMPLEX(f) * g0[8]
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# Multiply by scalar, real
@myjit
def mul_s_real(g0, f):
    r0 = GROUP_TYPE_REAL(f) * g0[0]
    r1 = GROUP_TYPE_REAL(f) * g0[1]
    r2 = GROUP_TYPE_REAL(f) * g0[2]
    r3 = GROUP_TYPE_REAL(f) * g0[3]
    r4 = GROUP_TYPE_REAL(f) * g0[4]
    r5 = GROUP_TYPE_REAL(f) * g0[5]
    r6 = GROUP_TYPE_REAL(f) * g0[6]
    r7 = GROUP_TYPE_REAL(f) * g0[7]
    r8 = GROUP_TYPE_REAL(f) * g0[8]
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

# Conjugate transpose
@myjit
def dagger(a):
    r0 = a[0].conjugate()
    r1 = a[1].conjugate()
    r2 = a[2].conjugate()
    r3 = a[3].conjugate()
    r4 = a[4].conjugate()
    r5 = a[5].conjugate()
    r6 = a[6].conjugate()
    r7 = a[7].conjugate()
    r8 = a[8].conjugate()
    
    return r0, r1, r2, r3, r4, r5, r6, r7, r8

"""
    Useful functions for temporary fields (setting to zero, unit and addition, ...)
"""

# Get algebra element zero
@myjit
def zero_algebra():
    return GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0)

# Get zero group element
@myjit
def zero():
    return (GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0))

#[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
@myjit
def init1():
    return (GROUP_TYPE_COMPLEX(- (1 / 3)), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
            GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(- (2 / math.sqrt(3.0))))

#[[1, 0, 0], [0, -1, 0], [0, 0, -1]]
@myjit
def init2():
    return (GROUP_TYPE_COMPLEX(- (1 / 3)), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(1.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
            GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(1 / math.sqrt(3.0)))

#[[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
@myjit
def init3():
    return (GROUP_TYPE_COMPLEX(- (1 / 3)), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(-1.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
            GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(1 / math.sqrt(3.0)))
            
# Get unit group element
@myjit
def unit():
    return (GROUP_TYPE_COMPLEX(1.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0),               GROUP_TYPE_COMPLEX(0.0))

# Group store: g0 <- g1
@myjit
def store(g_to, g_from):
    for i in range(len(g_from)):
        real_part = g_from[i].real
        imag_part = g_from[i].imag
        g_to[i] = GROUP_TYPE_COMPLEX(real_part + 1j * imag_part)

# Return tuple
@myjit
def load(g):
    return g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]

# Trace
@myjit
def tr(a):
    return 3 * a[0]

# Trace of square
@myjit
def sq(a):
    return 3 * (a[0] * a[0] + (2 / 3) * a[1] * a[1] + (2 / 3) * a[2] * a[2] + (2 / 3) * a[3] * a[3] + (4 / 6) * a[4] * a[4] + (4 / 6) * a[5] * a[5] + (2 / 3) * a[6] * a[6] 
                + (2 / 3) * a[7] * a[7] + (2 / 3) * a[8] * a[8])

# Normalize su(3) group element (Test before using)
@myjit
def normalize(u):
    u = to_matrix(u)
    array0, array1, array2, array3, array4, array5, array6, array7, array8 = (GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                                                              GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                                                              GROUP_TYPE_COMPLEX(0.0))
    array0 = u[0]
    array1 = u[1]
    array2 = u[2]
    array3 = u[3]
    array4 = u[4]
    array5 = u[5]
    array6 = u[6]
    array7 = u[7]
    array8 = u[8]
    
    m00, m01, m02 = array0, array1, array2
    m10, m11, m12 = array3, array4, array5
    m20, m21, m22 = array6, array7, array8
    
    # First column (v1)
    norm_v1 = math.sqrt(abs(m00)**2 + abs(m10)**2 + abs(m20)**2)
    u1_0 = m00 / norm_v1
    u1_1 = m10 / norm_v1
    u1_2 = m20 / norm_v1
    
    # Second column (v2) orthogonalized against u1
    dot_u1_v2 = (u1_0.conjugate() * m01 + 
                 u1_1.conjugate() * m11 + 
                 u1_2.conjugate() * m21)
    v2_0 = m01 - dot_u1_v2 * u1_0
    v2_1 = m11 - dot_u1_v2 * u1_1
    v2_2 = m21 - dot_u1_v2 * u1_2
    norm_v2 = math.sqrt(abs(v2_0)**2 + abs(v2_1)**2 + abs(v2_2)**2)
    u2_0 = v2_0 / norm_v2
    u2_1 = v2_1 / norm_v2
    u2_2 = v2_2 / norm_v2
    
    # Third column (v3) orthogonalized against u1 and u2
    dot_u1_v3 = (u1_0.conjugate() * m02 + 
                 u1_1.conjugate() * m12 + 
                 u1_2.conjugate() * m22)
    dot_u2_v3 = (u2_0.conjugate() * m02 + 
                 u2_1.conjugate() * m12 + 
                 u2_2.conjugate() * m22)
    v3_0 = m02 - dot_u1_v3 * u1_0 - dot_u2_v3 * u2_0
    v3_1 = m12 - dot_u1_v3 * u1_1 - dot_u2_v3 * u2_1
    v3_2 = m22 - dot_u1_v3 * u1_2 - dot_u2_v3 * u2_2
    norm_v3 = math.sqrt(abs(v3_0)**2 + abs(v3_1)**2 + abs(v3_2)**2)
    u3_0 = v3_0 / norm_v3
    u3_1 = v3_1 / norm_v3
    u3_2 = v3_2 / norm_v3
    
    # Ensure determinant = 1 by adjusting the phase of the last column
    det = (u1_0 * (u2_1 * u3_2 - u2_2 * u3_1) -
           u1_1 * (u2_0 * u3_2 - u2_2 * u3_0) +
           u1_2 * (u2_0 * u3_1 - u2_1 * u3_0))
    phase_correction = math.cos(-math.atan2(det.imag, det.real)) + 1j * math.sin(-math.atan2(det.imag, det.real))
    u3_0 *= phase_correction
    u3_1 *= phase_correction
    u3_2 *= phase_correction

    return to_repr((u1_0, u2_0, u3_0, u1_1, u2_1, u3_1, u1_2, u2_2, u3_2))

# Turn real-valued representation to 3x3 complex matrices
@myjit
def to_matrix(u):
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = (GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0),                                                                 GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0))
    
    r0 += GROUP_TYPE_COMPLEX(u[0] * id0[0])
    r1 += GROUP_TYPE_COMPLEX(u[0] * id0[1])
    r2 += GROUP_TYPE_COMPLEX(u[0] * id0[2])
    r3 += GROUP_TYPE_COMPLEX(u[0] * id0[3])
    r4 += GROUP_TYPE_COMPLEX(u[0] * id0[4])
    r5 += GROUP_TYPE_COMPLEX(u[0] * id0[5])
    r6 += GROUP_TYPE_COMPLEX(u[0] * id0[6])
    r7 += GROUP_TYPE_COMPLEX(u[0] * id0[7])
    r8 += GROUP_TYPE_COMPLEX(u[0] * id0[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[1] * gm1[0])
    r1 += GROUP_TYPE_COMPLEX(u[1] * gm1[1])
    r2 += GROUP_TYPE_COMPLEX(u[1] * gm1[2])
    r3 += GROUP_TYPE_COMPLEX(u[1] * gm1[3])
    r4 += GROUP_TYPE_COMPLEX(u[1] * gm1[4])
    r5 += GROUP_TYPE_COMPLEX(u[1] * gm1[5])
    r6 += GROUP_TYPE_COMPLEX(u[1] * gm1[6])
    r7 += GROUP_TYPE_COMPLEX(u[1] * gm1[7])
    r8 += GROUP_TYPE_COMPLEX(u[1] * gm1[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[2] * gm2[0])
    r1 += GROUP_TYPE_COMPLEX(u[2] * gm2[1])
    r2 += GROUP_TYPE_COMPLEX(u[2] * gm2[2])
    r3 += GROUP_TYPE_COMPLEX(u[2] * gm2[3])
    r4 += GROUP_TYPE_COMPLEX(u[2] * gm2[4])
    r5 += GROUP_TYPE_COMPLEX(u[2] * gm2[5])
    r6 += GROUP_TYPE_COMPLEX(u[2] * gm2[6])
    r7 += GROUP_TYPE_COMPLEX(u[2] * gm2[7])
    r8 += GROUP_TYPE_COMPLEX(u[2] * gm2[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[3] * gm3[0])
    r1 += GROUP_TYPE_COMPLEX(u[3] * gm3[1])
    r2 += GROUP_TYPE_COMPLEX(u[3] * gm3[2])
    r3 += GROUP_TYPE_COMPLEX(u[3] * gm3[3])
    r4 += GROUP_TYPE_COMPLEX(u[3] * gm3[4])
    r5 += GROUP_TYPE_COMPLEX(u[3] * gm3[5])
    r6 += GROUP_TYPE_COMPLEX(u[3] * gm3[6])
    r7 += GROUP_TYPE_COMPLEX(u[3] * gm3[7])
    r8 += GROUP_TYPE_COMPLEX(u[3] * gm3[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[4] * gm4[0])
    r1 += GROUP_TYPE_COMPLEX(u[4] * gm4[1])
    r2 += GROUP_TYPE_COMPLEX(u[4] * gm4[2])
    r3 += GROUP_TYPE_COMPLEX(u[4] * gm4[3])
    r4 += GROUP_TYPE_COMPLEX(u[4] * gm4[4])
    r5 += GROUP_TYPE_COMPLEX(u[4] * gm4[5])
    r6 += GROUP_TYPE_COMPLEX(u[4] * gm4[6])
    r7 += GROUP_TYPE_COMPLEX(u[4] * gm4[7])
    r8 += GROUP_TYPE_COMPLEX(u[4] * gm4[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[5] * gm5[0])
    r1 += GROUP_TYPE_COMPLEX(u[5] * gm5[1])
    r2 += GROUP_TYPE_COMPLEX(u[5] * gm5[2])
    r3 += GROUP_TYPE_COMPLEX(u[5] * gm5[3])
    r4 += GROUP_TYPE_COMPLEX(u[5] * gm5[4])
    r5 += GROUP_TYPE_COMPLEX(u[5] * gm5[5])
    r6 += GROUP_TYPE_COMPLEX(u[5] * gm5[6])
    r7 += GROUP_TYPE_COMPLEX(u[5] * gm5[7])
    r8 += GROUP_TYPE_COMPLEX(u[5] * gm5[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[6] * gm6[0])
    r1 += GROUP_TYPE_COMPLEX(u[6] * gm6[1])
    r2 += GROUP_TYPE_COMPLEX(u[6] * gm6[2])
    r3 += GROUP_TYPE_COMPLEX(u[6] * gm6[3])
    r4 += GROUP_TYPE_COMPLEX(u[6] * gm6[4])
    r5 += GROUP_TYPE_COMPLEX(u[6] * gm6[5])
    r6 += GROUP_TYPE_COMPLEX(u[6] * gm6[6])
    r7 += GROUP_TYPE_COMPLEX(u[6] * gm6[7])
    r8 += GROUP_TYPE_COMPLEX(u[6] * gm6[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[7] * gm7[0])
    r1 += GROUP_TYPE_COMPLEX(u[7] * gm7[1])
    r2 += GROUP_TYPE_COMPLEX(u[7] * gm7[2])
    r3 += GROUP_TYPE_COMPLEX(u[7] * gm7[3])
    r4 += GROUP_TYPE_COMPLEX(u[7] * gm7[4])
    r5 += GROUP_TYPE_COMPLEX(u[7] * gm7[5])
    r6 += GROUP_TYPE_COMPLEX(u[7] * gm7[6])
    r7 += GROUP_TYPE_COMPLEX(u[7] * gm7[7])
    r8 += GROUP_TYPE_COMPLEX(u[7] * gm7[8])
    
    r0 += GROUP_TYPE_COMPLEX(u[8] * gm8[0])
    r1 += GROUP_TYPE_COMPLEX(u[8] * gm8[1])
    r2 += GROUP_TYPE_COMPLEX(u[8] * gm8[2])
    r3 += GROUP_TYPE_COMPLEX(u[8] * gm8[3])
    r4 += GROUP_TYPE_COMPLEX(u[8] * gm8[4])
    r5 += GROUP_TYPE_COMPLEX(u[8] * gm8[5])
    r6 += GROUP_TYPE_COMPLEX(u[8] * gm8[6])
    r7 += GROUP_TYPE_COMPLEX(u[8] * gm8[7])
    r8 += GROUP_TYPE_COMPLEX(u[8] * gm8[8])
    
    r0img = (1 / 2) * (r0 - r0.conjugate()) / 1j
    r0real = (1 / 2) * (r0 + r0.conjugate())
    r1img = (1 / 2) * (r1 - r1.conjugate()) / 1j
    r1real = (1 / 2) * (r1 + r1.conjugate())
    r2img = (1 / 2) * (r2 - r2.conjugate()) / 1j
    r2real = (1 / 2) * (r2 + r2.conjugate())
    r3img = (1 / 2) * (r3 - r3.conjugate()) / 1j
    r3real = (1 / 2) * (r3 + r3.conjugate())
    r4img = (1 / 2) * (r4 - r4.conjugate()) / 1j
    r4real = (1 / 2) * (r4 + r4.conjugate())
    r5img = (1 / 2) * (r5 - r5.conjugate()) / 1j
    r5real = (1 / 2) * (r5 + r5.conjugate())
    r6img = (1 / 2) * (r6 - r6.conjugate()) / 1j
    r6real = (1 / 2) * (r6 + r6.conjugate())
    r7img = (1 / 2) * (r7 - r7.conjugate()) / 1j
    r7real = (1 / 2) * (r7 + r7.conjugate())
    r8img = (1 / 2) * (r8 - r8.conjugate()) / 1j
    r8real = (1 / 2) * (r8 + r8.conjugate())
        
    threshold2 = 5e-16
        
    if abs(r0img) < threshold2:
        r0img = 0.0
    if abs(r0real) < threshold2:
        r0real = 0.0
    if abs(r1img) < threshold2:
        r1img = 0.0
    if abs(r1real) < threshold2:
        r1real = 0.0
    if abs(r2img) < threshold2:
        r2img = 0.0
    if abs(r2real) < threshold2:
        r2real = 0.0
    if abs(r3img) < threshold2:
        r3img = 0.0
    if abs(r3real) < threshold2:
        r3real = 0.0
    if abs(r4img) < threshold2:
        r4img = 0.0
    if abs(r4real) < threshold2:
        r4real = 0.0
    if abs(r5img) < threshold2:
        r5img = 0.0
    if abs(r5real) < threshold2:
        r5real = 0.0
    if abs(r6img) < threshold2:
        r6img = 0.0
    if abs(r6real) < threshold2:
        r6real = 0.0
    if abs(r7img) < threshold2:
        r7img = 0.0
    if abs(r7real) < threshold2:
        r7real = 0.0
    if abs(r8img) < threshold2:
        r8img = 0.0
    if abs(r8real) < threshold2:
        r8real = 0.0
        
    ret0 = r0real + 1j * r0img
    ret1 = r1real + 1j * r1img
    ret2 = r2real + 1j * r2img
    ret3 = r3real + 1j * r3img
    ret4 = r4real + 1j * r4img
    ret5 = r5real + 1j * r5img
    ret6 = r6real + 1j * r6img
    ret7 = r7real + 1j * r7img
    ret8 = r8real + 1j * r8img
            
    return ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8
    
# Turn 3x3 complex matrices to complex-valued representation M = beta0 id0 + beta1 gm1 +...+ beta8 gm8
@myjit
def to_repr(m):
    beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8 = (GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                                                     GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), GROUP_TYPE_COMPLEX(0.0), 
                                                                     GROUP_TYPE_COMPLEX(0.0))
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = m
    
    # Initialize m_tl (traceless part) as a zero array
    m_tl0= GROUP_TYPE_COMPLEX(0.0)
    m_tl1= GROUP_TYPE_COMPLEX(0.0)
    m_tl2= GROUP_TYPE_COMPLEX(0.0)
    m_tl3= GROUP_TYPE_COMPLEX(0.0)
    m_tl4= GROUP_TYPE_COMPLEX(0.0)
    m_tl5= GROUP_TYPE_COMPLEX(0.0)
    m_tl6= GROUP_TYPE_COMPLEX(0.0)
    m_tl7= GROUP_TYPE_COMPLEX(0.0)
    m_tl8= GROUP_TYPE_COMPLEX(0.0)

    #beta0 = Tr(m)/3
    beta0 = ((m0 + m4 + m8) / 3)
    
    #Calculate m_traceless
    m_tl0 = m0 - beta0 * id0[0]
    m_tl1 = m1 - beta0 * id0[1]
    m_tl2 = m2 - beta0 * id0[2]
    m_tl3 = m3 - beta0 * id0[3]
    m_tl4 = m4 - beta0 * id0[4]
    m_tl5 = m5 - beta0 * id0[5]
    m_tl6 = m6 - beta0 * id0[6]
    m_tl7 = m7 - beta0 * id0[7]
    m_tl8 = m8 - beta0 * id0[8]
    
    #betaj = 1/2 Tr(m_tl * gmj)
    
    #calculate beta1 
    tr0gm1 = m_tl0 * gm1[0] + m_tl1 * gm1[3] + m_tl2 * gm1[6]
    tr4gm1 = m_tl3 * gm1[1] + m_tl4 * gm1[4] + m_tl5 * gm1[7]
    tr8gm1 = m_tl6 * gm1[2] + m_tl7 * gm1[5] + m_tl8 * gm1[8]
    
    beta1 = (1 / 2) * (tr0gm1 + tr4gm1 + tr8gm1)
    
    #calculate beta2 
    tr0gm2 = m_tl0 * gm2[0] + m_tl1 * gm2[3] + m_tl2 * gm2[6]
    tr4gm2 = m_tl3 * gm2[1] + m_tl4 * gm2[4] + m_tl5 * gm2[7]
    tr8gm2 = m_tl6 * gm2[2] + m_tl7 * gm2[5] + m_tl8 * gm2[8]
    
    beta2 = (1 / 2) * (tr0gm2 + tr4gm2 + tr8gm2)
    
    #calculate beta3 
    tr0gm3 = m_tl0 * gm3[0] + m_tl1 * gm3[3] + m_tl2 * gm3[6]
    tr4gm3 = m_tl3 * gm3[1] + m_tl4 * gm3[4] + m_tl5 * gm3[7]
    tr8gm3 = m_tl6 * gm3[2] + m_tl7 * gm3[5] + m_tl8 * gm3[8]
    
    beta3 = (1 / 2) * (tr0gm3 + tr4gm3 + tr8gm3)
    
    #calculate beta4 
    tr0gm4 = m_tl0 * gm4[0] + m_tl1 * gm4[3] + m_tl2 * gm4[6]
    tr4gm4 = m_tl3 * gm4[1] + m_tl4 * gm4[4] + m_tl5 * gm4[7]
    tr8gm4 = m_tl6 * gm4[2] + m_tl7 * gm4[5] + m_tl8 * gm4[8]
    
    beta4 = (1 / 2) * (tr0gm4 + tr4gm4 + tr8gm4)
    
    #calculate beta5 
    tr0gm5 = m_tl0 * gm5[0] + m_tl1 * gm5[3] + m_tl2 * gm5[6]
    tr4gm5 = m_tl3 * gm5[1] + m_tl4 * gm5[4] + m_tl5 * gm5[7]
    tr8gm5 = m_tl6 * gm5[2] + m_tl7 * gm5[5] + m_tl8 * gm5[8]
    
    beta5 = (1 / 2) * (tr0gm5 + tr4gm5 + tr8gm5)
    
    #calculate beta6
    tr0gm6 = m_tl0 * gm6[0] + m_tl1 * gm6[3] + m_tl2 * gm6[6]
    tr4gm6 = m_tl3 * gm6[1] + m_tl4 * gm6[4] + m_tl5 * gm6[7]
    tr8gm6 = m_tl6 * gm6[2] + m_tl7 * gm6[5] + m_tl8 * gm6[8]
    
    beta6 = (1 / 2) * (tr0gm6 + tr4gm6 + tr8gm6)
    
    #calculate beta7
    tr0gm7 = m_tl0 * gm7[0] + m_tl1 * gm7[3] + m_tl2 * gm7[6]
    tr4gm7 = m_tl3 * gm7[1] + m_tl4 * gm7[4] + m_tl5 * gm7[7]
    tr8gm7 = m_tl6 * gm7[2] + m_tl7 * gm7[5] + m_tl8 * gm7[8]
    
    beta7 = (1 / 2) * (tr0gm7 + tr4gm7 + tr8gm7)
    
    #calculate beta8
    tr0gm8 = m_tl0 * gm8[0] + m_tl1 * gm8[3] + m_tl2 * gm8[6]
    tr4gm8 = m_tl3 * gm8[1] + m_tl4 * gm8[4] + m_tl5 * gm8[7]
    tr8gm8 = m_tl6 * gm8[2] + m_tl7 * gm8[5] + m_tl8 * gm8[8]
    
    beta8 = (1 / 2) * (tr0gm8 + tr4gm8 + tr8gm8)
    
    beta0img = (1 / 2) * (beta0 - beta0.conjugate()) / 1j
    beta0real = (1 / 2) * (beta0 + beta0.conjugate())
    beta1img = (1 / 2) * (beta1 - beta1.conjugate()) / 1j
    beta1real = (1 / 2) * (beta1 + beta1.conjugate())
    beta2img = (1 / 2) * (beta2 - beta2.conjugate()) / 1j
    beta2real = (1 / 2) * (beta2 + beta2.conjugate())
    beta3img = (1 / 2) * (beta3 - beta3.conjugate()) / 1j
    beta3real = (1 / 2) * (beta3 + beta3.conjugate())
    beta4img = (1 / 2) * (beta4 - beta4.conjugate()) / 1j
    beta4real = (1 / 2) * (beta4 + beta4.conjugate())
    beta5img = (1 / 2) * (beta5 - beta5.conjugate()) / 1j
    beta5real = (1 / 2) * (beta5 + beta5.conjugate())
    beta6img = (1 / 2) * (beta6 - beta6.conjugate()) / 1j
    beta6real = (1 / 2) * (beta6 + beta6.conjugate())
    beta7img = (1 / 2) * (beta7 - beta7.conjugate()) / 1j
    beta7real = (1 / 2) * (beta7 + beta7.conjugate())
    beta8img = (1 / 2) * (beta8 - beta8.conjugate()) / 1j
    beta8real = (1 / 2) * (beta8 + beta8.conjugate())
    
    threshold = 5e-16
    
    if abs(beta0img) < threshold:
        beta0img = 0
    if abs(beta0real) < threshold:
        beta0real = 0
    if abs(beta1img) < threshold:
        beta1img = 0
    if abs(beta1real) < threshold:
        beta1real = 0
    if abs(beta2img) < threshold:
        beta2img = 0
    if abs(beta2real) < threshold:
        beta2real = 0
    if abs(beta3img) < threshold:
        beta3img = 0
    if abs(beta3real) < threshold:
        beta3real = 0
    if abs(beta4img) < threshold:
        beta4img = 0
    if abs(beta4real) < threshold:
        beta4real = 0
    if abs(beta5img) < threshold:
        beta5img = 0
    if abs(beta5real) < threshold:
        beta5real = 0
    if abs(beta6img) < threshold:
        beta6img = 0
    if abs(beta6real) < threshold:
        beta6real = 0
    if abs(beta7img) < threshold:
        beta7img = 0
    if abs(beta7real) < threshold:
        beta7real = 0
    if abs(beta8img) < threshold:
        beta8img = 0
    if abs(beta8real) < threshold:
        beta8real = 0

    ret0 = beta0real + 1j * beta0img
    ret1 = beta1real + 1j * beta1img
    ret2 = beta2real + 1j * beta2img
    ret3 = beta3real + 1j * beta3img
    ret4 = beta4real + 1j * beta4img
    ret5 = beta5real + 1j * beta5img
    ret6 = beta6real + 1j * beta6img
    ret7 = beta7real + 1j * beta7img
    ret8 = beta8real + 1j * beta8img
    
    return ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8
