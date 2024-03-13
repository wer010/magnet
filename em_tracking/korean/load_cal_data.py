#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
import copy


# In[2]:


data_raw = np.loadtxt("20240311_test3.txt")
print(np.shape(data_raw))
# data_raw = np.loadtxt("normal_data.txt")


# In[3]:


def cal_pos(matrix):
    r_a = matrix[9:12]
    r_b = matrix[12:15]
    r_c = matrix[15:18]
    t_a = matrix[18:21]
    t_b = matrix[21:24]
    t_c = matrix[24:27]
    r_d = r_d = (51.67396739673968 / 93.8176818448948 * (r_b - r_a)) + r_a
    t_d = (73.82978297829783 / 168.44277858900054 * (t_b - t_a)) + t_a
    receive_x = r_b - r_d
    receive_y = r_c - r_d
    transmit_x = t_b - t_d
    transmit_y = t_c - t_d

    r_axis_x = receive_x / np.linalg.norm(receive_x)
    r_axis_y = receive_y / np.linalg.norm(receive_y)
    r_axis_z = np.cross(r_axis_x, r_axis_y)
    r_axis = np.hstack((r_axis_x, r_axis_y))
    r_axis = np.hstack((r_axis, r_axis_z))
    r_axis = np.reshape(r_axis,(3,3))
    
    t_axis_x = transmit_x / np.linalg.norm(transmit_x)
    t_axis_y = transmit_y / np.linalg.norm(transmit_y)
    t_axis_z = np.cross(t_axis_x, t_axis_y)
    t_axis = np.hstack((t_axis_x, t_axis_y))
    t_axis = np.hstack((t_axis, t_axis_z))
    t_axis = np.reshape(t_axis,(3,3))
    relative_r = r_axis.dot(np.linalg.inv(t_axis))
#     relative_r = normalize_rotation_matrix(relative_r)
    pos = np.dot(t_axis, (r_d - t_d))/1000
#     matrix_r = R.from_matrix(relative_r)
#     euler = matrix_r.as_euler("xyz",degrees = True)
    relative_r = np.reshape(relative_r,(9,))
    return pos,relative_r


# In[4]:


r_all = []
for i in range(np.shape(data_raw)[0]):
# for i in range(3000,4000):
    pos,relative_r = cal_pos(data_raw[i])
    r = np.sqrt(np.sum(np.square(pos)))
    r_all.append(r)
    if i == 0:
        pos_all = pos
        relative_r_all = relative_r
        data = data_raw[i][0:9] 
    else:
        pos_all = np.vstack((pos_all,pos))
        relative_r_all = np.vstack((relative_r_all,relative_r))
        data = np.vstack((data,data_raw[i][0:9]))
r_all = np.array(r_all)


# In[5]:


from mpl_toolkits import mplot3d  
# Creating figures for the plot  
fig = plt.figure()  
ax = plt.axes(projection ="3d")  
ax.set_xlabel('X-axis', fontweight ='bold')  
ax.set_ylabel('Y-axis', fontweight ='bold')  
ax.set_zlabel('Z-axis', fontweight ='bold')  
ax.set_xlim([-0.5,0.5])
ax.set_ylim([-0.5,0.5])
ax.set_zlim([-0.5,0.5])
# Creating a plot using the random datasets   
# ax.plot(r_d[:,0], r_d[:,1], r_d[:,2], color = "red")  
ax.plot(pos_all[:,0], pos_all[:,1], pos_all[:,2], color = "red")  


plt.title("3D scatter plot")  
# display the  plot  
plt.show()  


# In[6]:


f1_x_factor = np.array([9.91799111e-10, -1.18536119e-11,  1.65072009e-11])
f1_y_factor = np.array([-1.40363327e-11,  7.76394898e-10,  3.99175667e-12])
f1_z_factor = np.array([-4.61721415e-12, -3.83052461e-13,  7.92748170e-10])

f2_x_factor = np.array([1.00495013e-09, -5.49719341e-12, -1.13997557e-11])
f2_y_factor = np.array([2.64988303e-11,  8.09614956e-10, -2.99294999e-11])
f2_z_factor = np.array([-2.99018981e-12,  2.18980608e-12,  8.41045631e-10])

f3_x_factor = np.array([1.03099047e-09, -2.22568015e-13, -6.35683494e-12])
f3_y_factor = np.array([-8.55517979e-12,  8.09060679e-10,  2.88504804e-13])
f3_z_factor = np.array([4.11595239e-12, 6.99139678e-12, 8.01764283e-10])


# In[7]:


data_temp = copy.deepcopy(data)
data_temp[:,0:3] = data[:,3:6]
data_temp[:,3:6] = data[:,6:9]
data_temp[:,6:9] = data[:,0:3]
data = copy.deepcopy(data_temp)


# In[8]:


b_f1_x_orth = data[:,0]*f1_x_factor[0] + data[:,1]*f1_x_factor[1] + data[:,2]*f1_x_factor[2]
b_f1_y_orth = data[:,0]*f1_y_factor[0] + data[:,1]*f1_y_factor[1] + data[:,2]*f1_y_factor[2]
b_f1_z_orth = data[:,0]*f1_z_factor[0] + data[:,1]*f1_z_factor[1] + data[:,2]*f1_z_factor[2]

b_f2_x_orth = data[:,3]*f2_x_factor[0] + data[:,4]*f2_x_factor[1] + data[:,5]*f2_x_factor[2]
b_f2_y_orth = data[:,3]*f2_y_factor[0] + data[:,4]*f2_y_factor[1] + data[:,5]*f2_y_factor[2]
b_f2_z_orth = data[:,3]*f2_z_factor[0] + data[:,4]*f2_z_factor[1] + data[:,5]*f2_z_factor[2]

b_f3_x_orth = data[:,6]*f3_x_factor[0] + data[:,7]*f3_x_factor[1] + data[:,8]*f3_x_factor[2]
b_f3_y_orth = data[:,6]*f3_y_factor[0] + data[:,7]*f3_y_factor[1] + data[:,8]*f3_y_factor[2]
b_f3_z_orth = data[:,6]*f3_z_factor[0] + data[:,7]*f3_z_factor[1] + data[:,8]*f3_z_factor[2]


# In[9]:


b_f1_x_orth = np.reshape(b_f1_x_orth,(np.shape(b_f1_x_orth)[0],1))
b_f1_y_orth = np.reshape(b_f1_y_orth,(np.shape(b_f1_x_orth)[0],1))
b_f1_z_orth = np.reshape(b_f1_z_orth,(np.shape(b_f1_x_orth)[0],1))
b_f2_x_orth = np.reshape(b_f2_x_orth,(np.shape(b_f1_x_orth)[0],1))
b_f2_y_orth = np.reshape(b_f2_y_orth,(np.shape(b_f1_x_orth)[0],1))
b_f2_z_orth = np.reshape(b_f2_z_orth,(np.shape(b_f1_x_orth)[0],1))

b_f3_x_orth = np.reshape(b_f3_x_orth,(np.shape(b_f1_x_orth)[0],1))
b_f3_y_orth = np.reshape(b_f3_y_orth,(np.shape(b_f1_x_orth)[0],1))
b_f3_z_orth = np.reshape(b_f3_z_orth,(np.shape(b_f1_x_orth)[0],1))


# In[10]:


b_orth = np.hstack((b_f1_x_orth,b_f1_y_orth))
b_orth = np.hstack((b_orth,b_f1_z_orth))
b_orth = np.hstack((b_orth,b_f2_x_orth))
b_orth = np.hstack((b_orth,b_f2_y_orth))
b_orth = np.hstack((b_orth,b_f2_z_orth))
b_orth = np.hstack((b_orth,b_f3_x_orth))
b_orth = np.hstack((b_orth,b_f3_y_orth))
b_orth = np.hstack((b_orth,b_f3_z_orth))
mea_mag = np.reshape(b_orth,(np.shape(b_orth)[0],3,3))
print(np.shape(b_orth))
mea_mag[0].T


# In[11]:


r_orth = []
f1_sum_list = []
f2_sum_list = []
f3_sum_list = []
por_x_all = []
theory_bt = 2.42e-9
for i in range(np.shape(b_orth)[0]):
    f1_sum = np.sum(np.square(b_orth[i,0:3]))
    f2_sum = np.sum(np.square(b_orth[i,3:6]))
    f3_sum = np.sum(np.square(b_orth[i,6:9]))
    f1_sum_list.append(f1_sum)
    f2_sum_list.append(f2_sum)
    f3_sum_list.append(f3_sum)
    por_x_all.append(f1_sum/(f1_sum + f2_sum + f3_sum))
    r_orth_temp = np.power(theory_bt*theory_bt*6/(f1_sum+f2_sum+f3_sum),1/6)
    r_orth.append(r_orth_temp)
    diff_x = f1_sum*(r_orth_temp**8)/(3*theory_bt*theory_bt) - r_orth_temp*r_orth_temp/3
    diff_y = f2_sum*(r_orth_temp**8)/(3*theory_bt*theory_bt) - r_orth_temp*r_orth_temp/3
    diff_z = f3_sum*(r_orth_temp**8)/(3*theory_bt*theory_bt) - r_orth_temp*r_orth_temp/3
    if diff_x < 0:
        x = -np.sqrt(-diff_x)
    else:
        x = np.sqrt(diff_x)
    if diff_y < 0:
        y = -np.sqrt(-diff_y)
    else:
        y = np.sqrt(diff_y)
    if diff_z < 0:
        z = -np.sqrt(-diff_z)
    else:
        z = np.sqrt(diff_z)
    if i==0:
        mea_pos = np.array([x,y,z])
    else:
        mea_pos = np.vstack((mea_pos,np.array([x,y,z])))
mea_pos *= np.array([1,-1,1])


# In[12]:


fig1 = plt.figure()
plt.plot(np.array(r_orth)*1000,"r")
plt.plot(np.array(r_all)*1000,"g")
plt.grid()
plt.show()


# In[13]:


fig1 = plt.figure()
plt.plot(mea_pos[:,0]*1000,"r",label = "x list")
plt.plot(pos_all[:,0]*1000,"g",label = "vicon x")
plt.grid()
plt.legend()
fig2 = plt.figure()
plt.plot(mea_pos[:,1]*1000,"r",label = "y list")
plt.plot(pos_all[:,1]*1000,"g",label = "vicon y")
plt.grid()
plt.legend()
fig3 = plt.figure()
plt.plot(mea_pos[:,2]*1000,"r",label = "z list")
plt.plot(pos_all[:,2]*1000,"g",label = "vicon z")
plt.grid()
plt.legend()
plt.show()


# In[14]:


euler_vicon = []
for i in range(np.shape(relative_r_all)[0]):
    matrix_ = relative_r_all[i].reshape(3,3)
    r_ = R.from_matrix(matrix_)
    euler_temp = r_.as_euler("xyz", degrees=True)
    euler_vicon.append(euler_temp)
euler_vicon = np.array(euler_vicon)


# In[15]:


fig1 = plt.figure()
plt.plot(euler_vicon[:,0],"r")
plt.plot(euler_vicon[:,1],"g")
plt.plot(euler_vicon[:,2],"b")
plt.grid()
plt.show()


# In[16]:


quaternion_rx = data_raw[:,39:43]
quaternion_tx = data_raw[:,43:47]
euler_imu = []
relative_imu = []
rotation_matrix = np.array([
    [np.cos(-np.pi), -np.sin(-np.pi), 0],
    [np.sin(-np.pi), np.cos(-np.pi), 0],
    [0, 0, 1]])
print(rotation_matrix)
# rotation_matrix = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]])
for i in range(np.shape(quaternion_rx)[0]):
    quat_rx_temp = R.from_quat(quaternion_rx[i])
    quat_tx_temp = R.from_quat(quaternion_tx[i])
    r_ = R.from_matrix(rotation_matrix)
    quat_rx_temp = r_*quat_rx_temp
    relative_imu_ = (quat_rx_temp*quat_tx_temp.inv())
    imu_euler_ = (relative_imu_).as_euler("zyx",degrees=True)
    r_ = R.from_euler("xyz",[imu_euler_[0],-imu_euler_[1],imu_euler_[2]],degrees = True)
    imu_euler_final = r_.as_euler("xyz",degrees=True)
    euler_imu.append(imu_euler_final)
    relative_imu.append(r_.as_matrix())
euler_imu = np.array(euler_imu)
relative_imu = np.array(relative_imu)


# In[17]:


fig2 = plt.figure()
plt.plot(euler_vicon[:,0],"r")
plt.plot(euler_imu[:,0],"g")
plt.grid()
plt.show()
fig2 = plt.figure()
plt.plot(euler_vicon[:,1],"r")
plt.plot(euler_imu[:,1],"g")
plt.grid()
plt.show()
fig2 = plt.figure()
plt.plot(euler_vicon[:,2],"r")
plt.plot(euler_imu[:,2],"g")
plt.grid()
plt.show()


# In[18]:


cal_mag = []
vicon_mag = []
vicon_mag_orth = []
for i in range(np.shape(mea_pos)[0]):
    r = np.sqrt(np.sum(np.square(mea_pos[i])))
    b_f1_x = theory_bt * (3 * (mea_pos[i][0] ** 2) / (r ** 5) - (1 / (r ** 3)))
    b_f1_y = theory_bt * 3 * mea_pos[i][0]*mea_pos[i][1] / (r**5)
    b_f1_z = theory_bt * 3 * mea_pos[i][0]*mea_pos[i][2] / (r**5)
    
    b_f2_x = theory_bt * 3 * mea_pos[i][0]*mea_pos[i][1] / (r**5)
    b_f2_y = theory_bt * (3 * (mea_pos[i][1] ** 2) / (r ** 5) - (1 / (r ** 3)))
    b_f2_z = theory_bt * 3 * mea_pos[i][1]*mea_pos[i][2] / (r**5)
    
    b_f3_x = theory_bt * 3 * mea_pos[i][0]*mea_pos[i][2] / (r**5)
    b_f3_y = theory_bt * 3 * mea_pos[i][1]*mea_pos[i][2] / (r**5)
    b_f3_z = theory_bt * (3 * (mea_pos[i][2] ** 2) / (r ** 5) - (1 / (r ** 3)))
    
    cal_mag.append([b_f1_x,b_f2_x,b_f3_x,b_f1_y,b_f2_y,b_f3_y,b_f1_z,b_f2_z,b_f3_z])
#     cal_mag.append([b_f1_x,b_f1_y,b_f1_z,b_f2_x,b_f2_y,b_f2_z,b_f3_x,b_f3_y,b_f3_z])
    r = np.sqrt(np.sum(np.square(pos_all[i])))
    b_f1_x_ = theory_bt * (3 * (pos_all[i][0] ** 2) / (r ** 5) - (1 / (r ** 3)))
    b_f1_y_ = theory_bt * 3 * pos_all[i][0]*pos_all[i][1] / (r**5)
    b_f1_z_ = theory_bt * 3 * pos_all[i][0]*pos_all[i][2] / (r**5)
    
    b_f2_x_ = theory_bt * 3 * pos_all[i][0]*pos_all[i][1] / (r**5)
    b_f2_y_ = theory_bt * (3 * (pos_all[i][1] ** 2) / (r ** 5) - (1 / (r ** 3)))
    b_f2_z_ = theory_bt * 3 * pos_all[i][1]*pos_all[i][2] / (r**5)
    
    b_f3_x_ = theory_bt * 3 * pos_all[i][0]*pos_all[i][2] / (r**5)
    b_f3_y_ = theory_bt * 3 * pos_all[i][1]*pos_all[i][2] / (r**5)
    b_f3_z_ = theory_bt * (3 * (pos_all[i][2] ** 2) / (r ** 5) - (1 / (r ** 3)))

    vicon_mag.append([b_f1_x_,b_f2_x_,b_f3_x_,b_f1_y_,b_f2_y_,b_f3_y_,b_f1_z_,b_f2_z_,b_f3_z_])
#     vicon_mag.append([b_f1_x_,b_f1_y_,b_f1_z_,b_f2_x_,b_f2_y_,b_f2_z_,b_f3_x_,b_f3_y_,b_f3_z_])
    relative_r = np.reshape(relative_r_all[i],(3,3))
    b_f1 = np.dot(relative_r,np.array([b_f1_x_,b_f1_y_,b_f1_z_]))
    b_f2 = np.dot(relative_r,np.array([b_f2_x_,b_f2_y_,b_f2_z_]))
    b_f3 = np.dot(relative_r,np.array([b_f3_x_,b_f3_y_,b_f3_z_]))
    vicon_mag_orth.append([b_f1[0],b_f2[0],b_f3[0],b_f1[1],b_f2[1],b_f3[1],b_f1[2],b_f2[2],b_f3[2]])
#     vicon_mag_orth.append([b_f1[0],b_f1[1],b_f1[2],b_f2[0],b_f2[1],b_f2[2],b_f3[0],b_f3[1],b_f3[2]])
cal_mag = np.array(cal_mag)
cal_mag = cal_mag.reshape(np.shape(cal_mag)[0],3,3)
vicon_mag = np.array(vicon_mag)
vicon_mag = np.reshape(vicon_mag,(np.shape(vicon_mag)[0],3,3))
vicon_mag_orth = np.array(vicon_mag_orth)
vicon_mag_orth = np.reshape(vicon_mag_orth,(np.shape(vicon_mag_orth)[0],3,3))


# In[19]:


# euler_mag = []
# mea_mag_symbol = []
# for i in range(np.shape(cal_mag)[0]):
#     cal_mag_matrix = np.dot(vicon_mag_orth[i],np.linalg.inv(vicon_mag[i]))
#     r_ = R.from_matrix(cal_mag_matrix)
#     euler = r_.as_euler("xyz",degrees=True)
#     euler_mag.append(euler)
# euler_mag = np.array(euler_mag)


# In[20]:


euler_mag = []
mea_mag_symbol = []
for i in range(np.shape(cal_mag)[0]):
    #定位到这一段，这一段的符号乘的有问题
    f1_symbole = np.dot(relative_imu[i],cal_mag[i])/np.abs(np.dot(relative_imu[i],cal_mag[i]))
#     f1_symbole = np.dot(relative_r_all[i].reshape(3,3),cal_mag[i])/np.abs(np.dot(relative_r_all[i].reshape(3,3),cal_mag[i]))
    mea_mag_symbol = f1_symbole*mea_mag[i].T
#     mea_mag_symbol.append(cal_mag_rx)
    cal_mag_matrix = np.dot(mea_mag_symbol,np.linalg.inv(cal_mag[i]))
    r_ = R.from_matrix(cal_mag_matrix)
    euler = r_.as_euler("xyz",degrees=True)
    euler_mag.append(euler)
euler_mag = np.array(euler_mag)
# mea_mag_symbol = np.array(mea_mag_symbol)


# In[21]:


# fig1 = plt.figure()
# # mea_mag_symbol = mea_mag_symbol.reshape(np.shape(mea_mag_symbol)[0],9)
# plt.plot(vicon_mag_orth[:,3])
# plt.plot(mea_mag_symbol[:,1,0])
# # plt.plot(vicon_mag[:,1])
# # plt.plot(cal_mag[:,1])
# # plt.plot(vicon_mag[:,2])
# # plt.plot(cal_mag[:,2])
# plt.grid()
# plt.show()


# In[25]:


fig1 = plt.figure()
plt.plot(euler_mag[:,1],"r")
plt.plot(euler_vicon[:,1],"g")

plt.grid()
plt.show()


# In[23]:


mea_pos


# In[24]:


r = np.array([[ 0.99440184, -0.05043613, -0.09285026],
 [ 0.09717751,  0.78157576,  0.61619467],
 [ 0.04149104, -0.62176807 , 0.78210163]])
a = np.array([[-2.32251772e-08, -1.58887073e-08, -2.06943292e-08],
 [-9.52732535e-09 , 2.60276603e-08, -2.48859740e-08],
 [ 5.74758974e-09, -2.84928797e-08, -8.74098193e-09]])
np.dot(r,a),np.dot(r,np.array([-2.32251772e-08, -9.52732535e-09, 5.74758974e-09]))


# In[ ]:




