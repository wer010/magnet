import numpy as np
import matplotlib.pyplot as plt
from rotation import Quaternion
from magnetic_field_models import Magnetic_dipole
from em_tracking.utils import get_octant_by_imu

sign_symbols = np.array([[1,1,1],
                         [-1,1,1],
                         [-1,-1,1],
                         [1,-1,1]])
pos0 = np.array([20,30,40])
r = np.linalg.norm(pos0)

b1t=1000
b2t=900
b3t=1100
m1=np.array([1,0,0])
m2=np.array([0,1,0])
m3=np.array([0,0,1])

m1 = Magnetic_dipole(b1t, m1)
m2 = Magnetic_dipole(b2t, m2)
m3 = Magnetic_dipole(b3t, m3)

b1_list = []
b2_list = []
b3_list = []
q= [np.cos(np.pi/15),np.sin(np.pi/15)/np.sqrt(3),np.sin(np.pi/15)/np.sqrt(3),np.sin(np.pi/15)/np.sqrt(3)]
q = Quaternion(*q)

b1_1o = m1.get_bvector(pos0)
b2_1o = m2.get_bvector(pos0)
b3_1o = m3.get_bvector(pos0)
b_1o = np.concatenate([b1_1o, b2_1o, b3_1o], axis=0)

for i in range(4):
    pos = pos0*sign_symbols[i]
    b1 = m1.get_bvector(pos)
    b2 = m2.get_bvector(pos)
    b3 = m3.get_bvector(pos)

    rm = q.q_to_r()
    b = np.concatenate([b1,b2,b3],axis=0)
    bs = rm@b.T
    oct = get_octant_by_imu(b_1o, np.abs(bs.T), rm)
    print(oct)




# oct = get_octant_by_imu(np.concatenate([b1,b2,b3]), np.abs(np.concatenate([b1s,b2s,b3s])), q.q_to_r())
# print(oct)