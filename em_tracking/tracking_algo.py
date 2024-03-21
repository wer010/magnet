import numpy as np

def get_pos_by_analytical_solution(b_sensor):
    theory_bt = 2.42e-9
    b1_norm = np.sum(b_sensor[:, 0:3] ** 2, axis=-1)
    b2_norm = np.sum(b_sensor[:, 3:6] ** 2, axis=-1)
    b3_norm = np.sum(b_sensor[:, 6:9] ** 2, axis=-1)
    r = np.power(theory_bt * theory_bt * 6 / (b1_norm + b2_norm + b3_norm), 1 / 6)
    x2 = b1_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    y2 = b2_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3
    z2 = b3_norm * (r ** 8) / (3 * theory_bt * theory_bt) - (r ** 2) / 3

    return np.stack([np.sqrt(np.abs(x2)),np.sqrt(np.abs(y2)),np.sqrt(np.abs(z2))],axis=-1),r

def get_pos_by_matrix_analysis(b_sensor):
    theory_bt = 2.42e-9
    y = b_sensor.reshape(-1,3,3).transpose([0,2,1])
    dm = np.diag([1,-1,-1])
    pos = []
    rm_list = []
    for i in range(y.shape[0]):
        u,s,v = np.linalg.svd(y[i])
        # p = np.power(np.sqrt(6) * theory_bt / np.linalg.norm(y[i]), 1 / 3) * v[0] # svd decomposition
        p = np.power(6 * theory_bt / np.sum(np.diag(s)*np.array([2,1,1])), 1 / 3) * v[0] # least square
        pos.append(p)
        rm = v@dm@u.T
        rm_list.append(rm)
    return np.stack(pos), np.linalg.norm(pos,axis=-1),np.stack(rm_list)

def get_octant_by_imu(b_a, bs_a, rm):
    '''
    3*3 tensor:param b_a: the magnetic vector in 1st octant. b1,b2,b3 are row vectors
    3*3 tensor:param bs_a: the absolute magnetic vector sensed by magnetic induced coils
    3*3 tensor:param rm: rotation matrix. bs = rm@b
    int:return: the index of octant (1-4)
    '''
    sign_b = [[[1,1,1],[1,1,1],[1,1,1]],
              [[1,-1,-1],[-1,1,1],[-1,1,1]],
              [[1,1,-1],[1,1,-1],[-1,-1,1]],
              [[1,-1,1],[-1,1,-1],[1,-1,1]]]

    e_list = []
    for i in range(4):
        b = b_a*sign_b[i]
        bs = np.abs(np.matmul(rm, b.T))
        e = bs.T - bs_a
        e_list.append(np.sum(e**2))
    e = np.concatenate([e_list])
    # print(e)
    oct = np.argmin(e)+1
    return oct
