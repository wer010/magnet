import matplotlib.pyplot as plt
import numpy as np
from rotation import Quaternion
sign_symbols = np.array([[1,1,1],
                         [-1,1,1],
                         [-1,-1,1],
                         [1,-1,1],
                         [1,1,-1],
                         [-1,1,-1],
                         [-1,-1,-1],
                         [1,-1,-1]])
color_list = ['tab:blue','tab:orange','tab:green','tab:red',
              'tab:purple','tab:brown','tab:pink','tab:gray']
pos0 = np.array([20,30,40])
r = np.linalg.norm(pos0)
b1t=1000
b2t=900
b3t=1100
m1=np.array([1,0,0])
m2=np.array([0,1,0])
m3=np.array([0,0,1])
b1_list = []
b2_list = []
b3_list = []
q= [np.cos(np.pi/15),np.sin(np.pi/15)/np.sqrt(3),np.sin(np.pi/15)/np.sqrt(3),np.sin(np.pi/15)/np.sqrt(3)]
q = Quaternion(*q)



for i in range(4):
    pos = pos0*sign_symbols[i]
    b1 = (b1t/(np.power(r,3)))*((3*(pos@m1)/np.power(r,2))*pos - m1)
    b2 = (b2t/(np.power(r,3)))*((3*(pos@m2)/np.power(r,2))*pos - m2)
    b3 = (b3t/(np.power(r,3)))*((3*(pos@m3)/np.power(r,2))*pos - m3)
    b1_list.append(b1)
    b2_list.append(b2)
    b3_list.append(b3)

    b1s = q.rotation(b1).squeeze()
    b2s = q.rotation(b2).squeeze()
    b3s = q.rotation(b3).squeeze()

    b1ss = np.abs(b1s)
    b2ss = np.abs(b2s)
    b3ss = np.abs(b3s)

    print(f'L2 norm of theoretical magnetic vectors are {np.linalg.norm(b1)}, {np.linalg.norm(b2)}, {np.linalg.norm(b3)}.')
    print(f'L2 norm of sensor magnetic vectors are {np.linalg.norm(b1s)}, {np.linalg.norm(b2s)}, {np.linalg.norm(b3s)}.')
    print(f'L2 norm of absolute sensor magnetic vectors are {np.linalg.norm(b1ss)}, {np.linalg.norm(b2ss)}, {np.linalg.norm(b3ss)}.')

    angle12 = b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2))
    angle23 = b2 @ b3 / (np.linalg.norm(b2) * np.linalg.norm(b3))
    angle31 = b3 @ b1 / (np.linalg.norm(b3) * np.linalg.norm(b1))
    v = np.cross(b1s, b2s) @ b3s
    print(f'In the sensor coordinate system, three angle of theoretical magnetic vectors is {angle12}, {angle23}, {angle31}, volume is {v}.')

    angle12 = b1s @ b2s / (np.linalg.norm(b1s) * np.linalg.norm(b2s))
    angle23 = b2s @ b3s / (np.linalg.norm(b2s) * np.linalg.norm(b3s))
    angle31 = b3s @ b1s / (np.linalg.norm(b3s) * np.linalg.norm(b1s))
    v = np.cross(b1s, b2s) @ b3s
    print(f'In the sensor coordinate system, three angle of sensor magnetic vectors is {angle12}, {angle23}, {angle31}, volume is {v}.')

    # angle12 = b1ss @ b2ss / (np.linalg.norm(b1ss) * np.linalg.norm(b2ss))
    # angle23 = b2ss @ b3ss / (np.linalg.norm(b2ss) * np.linalg.norm(b3ss))
    # angle31 = b3ss @ b1ss / (np.linalg.norm(b3ss) * np.linalg.norm(b1ss))
    # print(f'In the sensor coordinate system, three angle of absolute sensor magnetic vectors is {angle12}, {angle23}, {angle31}.')

    sign1 = b1s/b1ss
    sign2 = b2s/b2ss
    sign3 = b3s/b3ss
    print(np.stack([sign1,sign2,sign3]))
    n=0
    for s1 in sign_symbols:
        b1sg = b1ss*s1
        for s2 in sign_symbols:
            b2sg = b2ss*s2
            for s3 in sign_symbols:
                b3sg = b3ss*s3
                angle12g = b1sg @ b2sg / (np.linalg.norm(b1sg) * np.linalg.norm(b2sg))
                angle23g = b2sg @ b3sg / (np.linalg.norm(b2sg) * np.linalg.norm(b3sg))
                angle31g = b3sg @ b1sg / (np.linalg.norm(b3sg) * np.linalg.norm(b1sg))
                vg = np.cross(b1sg,b2sg)@b3sg
                if ([angle12g, angle23g, angle31g,vg] == [angle12, angle23, angle31,v]):
                    print(np.stack([s1,s2,s3]))
                    print(f'In the sensor coordinate system, three angle of guess sensor magnetic vectors is {angle12g}, {angle23g}, {angle31g}.')
                    n+=1
    print(n)