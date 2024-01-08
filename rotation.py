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
        ret = (self.w*self.w-self.r@self.r)*p + 2*(self.r@p)*self.r+2*self.w*np.cross(self.r,p)
        return ret

    def q_to_r(self):
        w = self.w
        a, b, c = self.r
        rs = [[w * w + a * a - b * b - c * c, 2 * a * b - 2 * c * w, 2 * a * c + 2 * b * w],
              [2 * a * b + 2 * w * c, w * w - a * a + b * b - c * c, 2 * b * c - 2 * a * w],
              [2 * a * c - 2 * b * w, 2 * b * c + 2 * a * w, w * w - a * a - b * b + c * c]]
        # r = R.from_quat([a,b,r,w])
        # print(r.as_matrix())
        # print(np.isclose(rs,r.as_matrix(),1e-5))

        return rs


if __name__ == '__main__':

    q= [np.cos(np.pi/6),np.sin(np.pi/6),0,0]

    r = R.from_quat([q[1],q[2],q[3],q[0]])

    q= Quaternion(q[0],q[1],q[2],q[3])
    print(q.get_q())
    v = np.array([0,1,0])

    v1 = r.as_matrix()@v
    v2 = q.rotation(v)

    print(np.isclose(v1,v2))
    print(np.isclose(q.q_to_r(),r.as_matrix()))