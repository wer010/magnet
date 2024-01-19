# TODO: the entrance for the algorithm testing
import numpy as np
from magnetic_field_models import Magnetic_datadriven,Magnetic_dipole
from rotation import Quaternion
from LM import lm


class Func:
    def __init__(self,p,model):
        # arg p means parameters, self.p means position
        self.set_p(p)
        self.model = model
        self.num_iter=0

    def forward(self,x):
        p_s_array = self.r.rotation(x) + self.p
        b_gt_array = self.model.get_bvector(p_s_array)
        b_s_array = np.linalg.inv(self.r.q_to_r()) @ b_gt_array

        self.num_iter+=1
        return b_s_array

    def set_p(self,p):
        self.p = p[0:3]
        self.r = Quaternion(p[3], p[4], p[5], p[6])

    def get_J(self, x, y, dp=0.001, update = 'oneside'):

        #update ： 'oneside','center' or 'zero'
        # number of data points
        m = len(y)
        # number of parameters
        n = len(self.p)

        # initialize Jacobian to Zero
        pc = self.p
        J = np.zeros((m, n))

        dp = dp * (1 + abs(self.p))
        # START --- loop over all parameters
        if update is not 'zero':
            for j in range(n):
                mask = np.zeros(n)
                mask[j]=1

                self.set_p(pc + dp*mask)
                y1 = self.forward(x)

                if update is 'oneside':
                    # J_ij = f(x+dx)-f(x)/dx
                    J[:, j] = (y1 - y) / dp[j]
                else:
                    # J_ij = f(x+dx)-f(x-dx)/2*dx
                    self.set_p(pc - dp * mask)
                    y2 = self.forward(x)
                    J[:, j] = (y1 - y2) / (2 * dp[j])
            self.set_p(pc)
        return J



def main():
    # 1. Load the ground truth data and initialize the model
    # mf_gt = Magnetic_datadriven(1e5,np.array([0,0,1]),'/home/lanhai/restore/dataset/magnetic tracking/output.fld')
    mf_dipole_model = Magnetic_dipole(1e5,np.array([0,0,1]))
    # 2. Give some points, get the magnetic induction intensity
    p = np.array([0.2,0.2,0.2])
    b_gt = mf_dipole_model.get_bvector(p)
    # 3. now the ground truth data is ready, let's define a sensor array and simulate of sensor data
    s_array = np.array([[0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0]])
    # 4. the coordinate axis of sensor array is different from the world coordinate axis.
    # we only consider the rotation. quaternion is used to denote the rotation.
    q = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0, 0]
    q = Quaternion(q[0], q[1], q[2], q[3])
    # 5. get the position of every sensor, then get the magnetic vector corresponding to the positions
    # use quaternion to get sensor data along each sensor axis
    p_s_array = q.rotation(s_array)+p
    b_gt_array = mf_dipole_model.get_bvector(p_s_array)
    b_s_array = np.linalg.inv(q.q_to_r())@b_gt_array
    # TODO 6. use numerical method to solve the position p from sensor array data.
    lm_func = Func(p=p, model = mf_dipole_model)

    p_c,_ = lm([0,0,0, q[0], q[1], q[2], q[3]], s_array, b_s_array, lm_func)

    # 7. assess and visualize the error
    print(p-p_c)


    return 0




if __name__ == '__main__':
    main()



