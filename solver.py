# TODO: implement the numerial solve via LM algorithm
import numpy as np
import torch
class Func:
    # 2D xz plane magnetic dipole model
    def __init__(self,p):
        # p means 2D position [x,z] of sensor array
        # r means 2D rotation [r] of sensor array
        self.set_p(p)
        self.num_iter=0

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


    def forward(self,x):
        # x \in R^{n,2}
        pos = self.p[0:2]
        rot = self.p[-1]
        x= pos + x
        bt = 1000
        r = np.linalg.norm(x,axis=-1)
        bx = (bt/np.power(r,5)) * (x[:,0]*x[:,1])
        bz = (bt/np.power(r,5)) * (3*x[:,1]*x[:,1]-np.power(r,2))
        bi = np.cos(rot)*bx+np.sin(rot)*bz
        bk = -np.sin(rot)*bx+np.cos(rot)*bz
        self.num_iter+=1
        return np.concatenate([bi,bk])   #return result \in R^n*2

    def set_p(self,p):
        self.p = p

    def get_Jacobian(self, x, y, dp=0.001, update='oneside'):

        # update ： 'oneside','center' or 'zero'
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
                mask[j] = 1

                self.set_p(pc + dp * mask)
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

    def get_Broyden_Jacobian(self,p_old,y_old,J,p,y):
        h = p - p_old

        a = (np.array([y - y_old]).T - J @ h) @ h.T
        b = h.T @ h

        # Broyden rank-1 update eq'n
        J = J + a / b

        return J

    def get_Hessian(self,**kwargs):
        J = self.get_Jacobian()
        H = J.T@J
        return H


class optim:
    def __init__(self, p_init, x, y, func, max_iter=1000):
        self.max_iter = max_iter
        self.func = func
        self.p = p_init
        self.x = x
        self.y = y
        self.Npar = len(self.p)
        self.Npnt = len(self.y)
        self.p_min = -1000
        self.p_max = 1000

    def loss(self,y_hat):
        # loss = \sum (\bar{y} - y)^2
        delta_y = (self.y - y_hat).reshape(-1, 1)
        X2 = delta_y.T @ delta_y
        return delta_y,X2

    def __call__(self):
        return self.op()

    def op(self):
        raise NotImplementedError


class Levenberg_Marquardt(optim):

    def __init__(self,*args):
        super().__init__(*args)
        self.weight = 1 / (self.y.T @ self.y)
        self.W = np.eye(self.Npnt) * self.weight
        self.DoF = self.Npnt - self.Npar + 1
        self.epsilon_1 = 1e-6  # convergence tolerance for gradient
        self.epsilon_2 = 1e-6  # convergence tolerance for parameters
        self.epsilon_4 = 1e-5  # determines acceptance of a L-M step
        self.lambda_0 = 1e-2  # initial value of damping paramter, lambda
        self.lambda_UP_fac = 11  # factor for increasing lambda
        self.lambda_DN_fac = 9
        self.use_broyden= False

    def loss(self,y_hat):
        delta_y = (self.y - y_hat).reshape(-1, 1)
        chi_sq = delta_y.T @ (self.W @ delta_y)
        return delta_y, chi_sq

    def op(self):
        cvg_hst=[]
        iteration = 0
        while iteration < self.max_iter:

            self.func.set_p(self.p)
            # evaluate model using parameters 'p'
            y_hat = self.func(self.x)
            delta_y, chi_sq = self.loss(y_hat)
            # TODO
            if self.use_broyden:
                if iteration % (2 * self.Npar) == 0 or dX2 > 0:
                    # finite difference
                    J = self.func.get_Jacobian(self.x, y_hat)
                else:
                    # rank-1 update
                    J = lm_Broyden_J(p_old, y_old, J, p, y_hat)
            else:
                J = self.func.get_Jacobian(self.x, y_hat)

            JtWJ = J.T @ (self.W @ J)
            JtWdy = J.T @ (self.W @ delta_y)

            if np.abs(JtWdy).max() < self.epsilon_1:
                print('*** Your Initial Guess is Extremely Close to Optimal ***')
                print('**** Convergence in r.h.s. ("JtWdy")  ****')
                break
            # incremental change in parameters
            # Marquardt

            h = np.linalg.solve((JtWJ + self.lambda_0 * np.diag(np.diag(JtWJ))), JtWdy)

            p_try = self.p + h.squeeze()
            # apply constraints
            p_try = np.minimum(np.maximum(self.p_min, p_try), self.p_max)

            # residual error using p_try
            self.func.set_p(p_try)
            y_hat_try = self.func(self.x)
            delta_y_try, chi_sq_try = self.loss(y_hat_try)

            # floating point error; break
            if not all(np.isfinite(delta_y)):
                break

            rho = (chi_sq - chi_sq_try) / np.abs(h.T @ (self.lambda_0 * np.diag(np.diag(JtWJ)) @ h + JtWdy))

            cvg_hst.append({'X2': chi_sq, 'p': self.p, 'lambda': self.lambda_0, 'y_hat': y_hat})

            # it IS significantly better
            if (rho > self.epsilon_4):
                # % accept p_try
                self.p = p_try
                self.lambda_0 = max(self.lambda_0 / self.lambda_DN_fac, 1.e-7)
            else:
                # % increase lambda  ==> gradient descent method
                # % Levenberg
                self.lambda_0 = min(self.lambda_0 * self.lambda_UP_fac, 1.e7)

            # update convergence history ... save _reduced_ Chi-square

            if (np.max(np.abs(h.squeeze()) / (np.abs(self.p) + 1e-12)) < self.epsilon_2 and iteration > 2):
                print('**** Convergence in Parameters ****')
                break

            iteration = iteration + 1
        return self.p, cvg_hst
    # TODO
    def error_analysis(self):
        pass


class Gradient_Descent(optim):
    def __init__(self, *args):
        super().__init__(*args)
        self.lambda_0 = 5e-1

    def op(self):
        cvg_hst = []
        iteration = 0
        while iteration < self.max_iter:
            self.func.set_p(self.p)
            y_hat = self.func(self.x)
            delta_y, chi_sq = self.loss(y_hat)
            J = self.func.get_Jacobian(self.x, y_hat)
            self.p = self.p + self.lambda_0* (J.T@delta_y).squeeze()
            cvg_hst.append({'X2': chi_sq , 'p': self.p, 'lambda': self.lambda_0, 'y_hat': y_hat})
            iteration+=1
        return self.p, cvg_hst



class Gauss_Newton(optim):
    def __init__(self, *args):
        super().__init__(*args)

    def op(self):
        cvg_hst = []
        iteration = 0
        while iteration < self.max_iter:
            self.func.set_p(self.p)
            y_hat = self.func(self.x)
            delta_y, chi_sq = self.loss(y_hat)
            J = self.func.get_Jacobian(self.x, y_hat)
            self.p = self.p + np.linalg.inv(J.T@J) @ (J.T @ delta_y).squeeze()
            cvg_hst.append({'X2': chi_sq, 'p': self.p,  'y_hat': y_hat})
            iteration += 1
        return self.p, cvg_hst


