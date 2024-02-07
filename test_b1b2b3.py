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
q= [np.cos(np.pi/4),np.sin(np.pi/4),0,0]
q = Quaternion(*q)



for i in range(4):
    pos = pos0*sign_symbols[i]
    b1 = (b1t/(np.power(r,3)))*((3*(pos@m1)/np.power(r,2))*pos - m1)
    b2 = (b2t/(np.power(r,3)))*((3*(pos@m2)/np.power(r,2))*pos - m2)
    b3 = (b3t/(np.power(r,3)))*((3*(pos@m3)/np.power(r,2))*pos - m3)
    b1_list.append(b1)
    b2_list.append(b2)
    b3_list.append(b3)


print(np.stack([b1_list]))
print(np.stack([b2_list]))
print(np.stack([b3_list]))



b1 = (b1t / (np.power(r, 3))) * ((3 * (pos0 @ m1) / np.power(r, 2)) * pos0 - m1)
b2 = (b2t / (np.power(r, 3))) * ((3 * (pos0 @ m2) / np.power(r, 2)) * pos0 - m2)
b3 = (b3t / (np.power(r, 3))) * ((3 * (pos0 @ m3) / np.power(r, 2)) * pos0 - m3)
b1s = q.rotation(b1).squeeze()
b2s = q.rotation(b2).squeeze()
b3s = q.rotation(b3).squeeze()
print(f'{b1s}, {b2s}, {b3s}')
print(f'L2 norm of sensor magnetic vectors are {np.linalg.norm(b1s)}, {np.linalg.norm(b2s)}, {np.linalg.norm(b3s)}.')
angle12 = b1s@b2s/(np.linalg.norm(b1s)*np.linalg.norm(b2s))
angle23 = b2s@b3s/(np.linalg.norm(b2s)*np.linalg.norm(b3s))
angle31 = b3s@b1s/(np.linalg.norm(b3s)*np.linalg.norm(b1s))
print(f'In the sensor coordinate system, three angle of magnetic vectors is {angle12}, {angle23}, {angle31}.')


fig = plt.figure(figsize=(12, 5))
for i in range(len(b1_list)):
    ax = fig.add_subplot(241+i,projection='3d')
    b1n = np.linalg.norm(b1_list[i])
    b2n = np.linalg.norm(b2_list[i])
    b3n = np.linalg.norm(b3_list[i])

    b1s = q.rotation(b1_list[i]).squeeze()
    b2s = q.rotation(b2_list[i]).squeeze()
    b3s = q.rotation(b3_list[i]).squeeze()
    print(f'In the {i}th octant, b1s is {b1s}, b2s is {b2s}, b3s is {b3s}.')
    # print(f'L2 norm of magnetic vectors are {b1n}, {b2n}, {b3n}.')
    ax.quiver(0, 0, 0, b1_list[i][0], b1_list[i][1], b1_list[i][2], color='red' ,length = 100*b1n/max(b1n,b2n,b3n))
    ax.quiver(0, 0, 0, b2_list[i][0], b2_list[i][1], b2_list[i][2], color='green' ,length = 100*b2n/max(b1n,b2n,b3n))
    ax.quiver(0, 0, 0, b3_list[i][0], b3_list[i][1], b3_list[i][2], color='blue' ,length = 100*b3n/max(b1n,b2n,b3n))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    angle12 = b1_list[i]@b2_list[i]/(b1n*b2n)
    angle23 = b2_list[i]@b3_list[i]/(b2n*b3n)
    angle31 = b3_list[i]@b1_list[i]/(b3n*b1n)
    # print(f'In the {i}th octant, three angle of magnetic vectors is {angle12}, {angle23}, {angle31}.')
    # print(f'The volume of three magnetic vector is {np.cross(b1_list[i],b2_list[i])@b3_list[i]}')


plt.show()

