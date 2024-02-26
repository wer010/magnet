#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# %matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq


# In[32]:


data_raw = np.loadtxt("duijiaoxian.txt")


# In[33]:


def cal_pos(matrix):
    offset = 0
    r_a = matrix[9-offset:12-offset]
    r_b = matrix[12-offset:15-offset]
    r_c = matrix[15-offset:18-offset]
    t_a = matrix[18-offset:21-offset]
    t_b = matrix[21-offset:24-offset]
    t_c = matrix[24-offset:27-offset]
    r_d = (51.67396739673968 / 93.8176818448948 * (r_b - r_a)) + r_a
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


# In[34]:

r_all = []
for i in range(np.shape(data_raw)[0]):
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


# In[35]:


mse_relative_acc = np.zeros([100,3])
mse_array = []
#应该是计算加速度的相对变化量
for i in range(np.shape(data_raw)[0]):
    mse_relative_acc[:-1] = mse_relative_acc[1:]
#     diff = np.array([np.abs((data_raw[i,0] - data_raw[i,3])/data_raw[i,3]), 
#                      np.abs((data_raw[i,1] - data_raw[i,4])/data_raw[i,4]), 
#                      np.abs((data_raw[i,2] - data_raw[i,5])/data_raw[i,5])])
    diff = np.array([np.abs((data_raw[i,0] - data_raw[i,3])/1.), 
                     np.abs((data_raw[i,1] - data_raw[i,4])/1.), 
                     np.abs((data_raw[i,2] - data_raw[i,5])/1.)])
    mse_relative_acc[-1] = diff
    mse_temp = np.sum(mse_relative_acc, axis = 0)
#     print(mse_temp)
#     print("--------")
    mse_array.append(mse_temp)
mse_array = np.array(mse_array)/200


# In[36]:


#计算斜率
gap = 2
slope_imu = []
slope_mag = []
for i in range(3,np.shape(data_raw)[0]-3):
    slope_x = np.array([gap*2+1,data_raw[i+3,0] - data_raw[i-3,0]])
    slope_y = np.array([gap*2+1,data_raw[i+3,1] - data_raw[i-3,0]])
    slope_z = np.array([gap*2+1,data_raw[i+3,2] - data_raw[i-3,0]])
    slope_imu.append(np.array([slope_x,slope_y,slope_z]))
    slope_x = np.array([gap*2+1,data_raw[i+3,3] - data_raw[i-3,3]])
    slope_y = np.array([gap*2+1,data_raw[i+3,4] - data_raw[i-3,4]])
    slope_z = np.array([gap*2+1,data_raw[i+3,5] - data_raw[i-3,5]])
    slope_mag.append(np.array([slope_x,slope_y,slope_z]))
slope_imu = np.array(slope_imu)
slope_mag = np.array(slope_mag)
np.shape(slope_imu)


# In[37]:


from mpl_toolkits import mplot3d  
# Creating figures for the plot  
fig = plt.figure()  
ax = plt.axes(projection ="3d")  
ax.set_xlabel('X-axis', fontweight ='bold')  
ax.set_ylabel('Y-axis', fontweight ='bold')  
ax.set_zlabel('Z-axis', fontweight ='bold')  
# ax.set_xlim([-0.5,0.5])
# ax.set_ylim([-0.5,0.5])
# ax.set_zlim([-0.5,0.5])
# Creating a plot using the random datasets   
# ax.plot(r_d[:,0], r_d[:,1], r_d[:,2], color = "red")  
ax.plot(pos_all[:,0], pos_all[:,1], pos_all[:,2], color = "red")  


plt.title("3D scatter plot")  
# display the  plot  
plt.show()  


# In[8]:


diff = []
thread = 10
for i in range(np.shape(data_raw)[0]):
    if np.abs(data_raw[i,3])<thread:
        diff.append(0)
    else:
        diff_ = np.abs(data_raw[i,3] - data_raw[i,0])
        if diff_ >=thread:
            diff.append(thread)
        else:
            diff.append(-thread)


# In[9]:


#可以考虑用斜率的方式来判断
#使用向量的夹角，就是间隔5到10个点，首先两个加速的绝对值要相近，然后加速度绝对值要大于10.然后判断中间间隔10个点计算一个向量，求向量的夹角.
#一直计算隔10个点的斜率
#两个imu之间也要去找修正的角度差  
fig1 = plt.figure()
plt.plot(data_raw[:,0],"r",label = "imu x")
plt.plot(data_raw[:,3],"g",label = "mag x")
plt.plot(diff,"c",label="diff")
# plt.plot(np.abs((data_raw[:,0]-data_raw[:,3])/data[:,3]))
# plt.plot(slope_imu[:,0,1],"r",label = "slope imu x")
# plt.plot(slope_mag[:,0,1],"g",label = "slope mag x")

plt.plot(pos_all[:,0]*50,"b",label = "pos x")
# plt.plot(mse_array[:,0],"c",label = "mse x")

plt.grid()
plt.legend()
plt.show()


# In[10]:


diff = []
thread = 18
for i in range(np.shape(data_raw)[0]):
    if np.abs(data_raw[i,4])<thread:
        diff.append(0)
    else:
        diff_ = np.abs(data_raw[i,4] - data_raw[i,1])
        if diff_ >=thread:
            diff.append(thread)
        else:
            diff.append(-thread)


# In[11]:


fig1 = plt.figure()
plt.plot(data_raw[:,1],"r",label = "imu y")
plt.plot(data_raw[:,4],"g",label = "mag y")
plt.plot(pos_all[:,1]*50,"b",label = "pos y")
plt.plot(diff,"c",label = "diff")

# plt.plot(mse_array[:,1],"c",label = "mse y")
# plt.plot(slope_imu[:,1,1],"r",label = "slope imu y")
# plt.plot(slope_mag[:,1,1],"g",label = "slope mag y")

plt.grid()
plt.legend()
plt.show()


# In[12]:


diff = []
thread = 20
for i in range(np.shape(data_raw)[0]):
    if np.abs(data_raw[i,5])<thread:
        diff.append(0)
    else:
        diff_ = np.abs(data_raw[i,5] - data_raw[i,2])
        if diff_ >=thread:
            diff.append(thread)
        else:
            diff.append(-thread)


# In[13]:


fig1 = plt.figure()
plt.plot(data_raw[:,2],"r",label = "imu z")
plt.plot(data_raw[:,5],"g",label = "mag z")
plt.plot(diff,"c",label = "diff")
plt.plot(pos_all[:,2]*50,"b",label = "pos z")
# plt.plot(mse_array[:,2],"c",label = "mse z")

plt.grid()
plt.legend()
plt.show()


# In[14]:


# #就算曲率
# import numpy as np

# # 假设有一系列点的坐标
# x = np.linspace(start = 0,stop = np.shape(pos_all)[0]-1,num = np.shape(pos_all)[0])  # 点的 x 坐标
# y = data_raw[:,0]  # 点的 y 坐标

# # 计算相邻点之间的方向向量
# dx = np.diff(x)
# dy = np.diff(y)

# # 计算相邻方向向量的夹角
# dot_products = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
# norms = np.sqrt(dx[:-1]**2 + dy[:-1]**2) * np.sqrt(dx[1:]**2 + dy[1:]**2)
# cos_theta = dot_products / norms
# theta = np.arccos(cos_theta)

# # 计算离散曲线的曲率
# curvature2 = 2 * np.sin(theta / 2) / norms

