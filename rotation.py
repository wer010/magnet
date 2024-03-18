import numpy as np
from scipy.spatial.transform import Rotation as R

class Quaternion:
    def __init__(self,w,a,b,c):
        self.set_q(w,a,b,c)

    def set_q(self,w,a,b,c):
        self.w = w
        self.r = np.array([a, b, c])

    def get_q(self):
        return np.concatenate([[self.w],self.r])

    def rotation(self,p):
        # ret = (self.w*self.w-self.r@self.r)*p + 2*(self.r@p)*self.r + 2*self.w*np.cross(self.r,p)
        ret = (self.w*self.w-self.r@self.r)*p + 2*(p@self.r).reshape(-1,1)*self.r.reshape(1,-1) + 2*self.w*np.cross(self.r,p)

        return ret

    def q_to_r(self):
        w = self.w
        a, b, c = self.r
        rs = np.array([[w * w + a * a - b * b - c * c, 2 * a * b - 2 * c * w, 2 * a * c + 2 * b * w],
                       [2 * a * b + 2 * w * c, w * w - a * a + b * b - c * c, 2 * b * c - 2 * a * w],
                       [2 * a * c - 2 * b * w, 2 * b * c + 2 * a * w, w * w - a * a - b * b + c * c]])
        # r = R.from_quat([a,b,r,w])
        # print(r.as_matrix())
        # print(np.isclose(rs,r.as_matrix(),1e-5))

        return rs


if __name__ == '__main__':

    theta = np.pi/4
    q = [np.cos(theta),np.sin(theta)*np.sqrt(3)/3,np.sin(theta)*np.sqrt(3)/3,np.sin(theta)*np.sqrt(3)/3]

    r = R.from_quat([q[1],q[2],q[3],q[0]])

    q= Quaternion(*q)
    print(q.get_q())
    v = np.array([0,1,0])

    v1 = r.as_matrix()@v
    v2 = q.rotation(v)
    print(v1,v2)
    print(np.isclose(v1,v2))

    q_star = [np.cos(theta),-np.sin(theta)*np.sqrt(3)/3,-np.sin(theta)*np.sqrt(3)/3,-np.sin(theta)*np.sqrt(3)/3]
    q_star = Quaternion(*q_star)
    v3 = q_star.rotation(v2)
    print('inverse rotation')
    print(np.isclose(v, v3))


    v = np.tile(v,(4,1))
    v1 = (r.as_matrix()@v.T).T
    v2 = q.rotation(v)
    print(v1,v2)
    print(np.isclose(v1,v2))
    # print(np.isclose(q.q_to_r(),r.as_matrix()))