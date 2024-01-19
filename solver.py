# TODO: implement the numerial solve via LM algorithm
import numpy as np
import torch
torch.nn.Conv2d

class optim:
    def __init__(self, p_init, x, y, func, max_iter=1000):
        self.max_iter= max_iter
        self.func = func
        self.p = p_init
        self.x = x
        self.y = y


    def loss(self):
        pass

    def update(self):
        return

    def __call__(self, **kwargs):
        self.op(**kwargs)

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


    def op(self):
        raise NotImplementedError


class Levenberg_Marquardt(optim):

    def loss(self,p):

        y_hat = lm_func.forward(x)
        X2 = delta_y.T @ (W @ delta_y)


    def op(self):
        while iteration < MaxIter:

            lm_func.set_p(p)
            # evaluate model using parameters 'p'
            y_hat = lm_func.forward(x)

            # TODO
            if use_broyden:
                if iteration % (2 * Npar) == 0 or dX2 > 0:
                    # finite difference
                    J = lm_func.get_J(x, y_hat)
                else:
                    # rank-1 update
                    J = lm_Broyden_J(p_old, y_old, J, p, y_hat)
            else:
                J = lm_func.get_J(x, y_hat)

            # residual error between model and data
            delta_y = (y - y_hat).reshape(-1, 1)

            # Chi-squared error criteria
            X2 = delta_y.T @ (W @ delta_y)
            JtWJ = J.T @ (W @ J)
            JtWdy = J.T @ (W @ delta_y)

            if np.abs(JtWdy).max() < epsilon_1:
                print('*** Your Initial Guess is Extremely Close to Optimal ***')
                print('**** Convergence in r.h.s. ("JtWdy")  ****')
                break
            # incremental change in parameters
            # Marquardt
            if Update_Type == 1:
                h = np.linalg.solve((JtWJ + lambda_ * np.diag(np.diag(JtWJ))), JtWdy)
            # Quadratic and Nielsen
            else:
                h = np.linalg.solve((JtWJ + lambda_ * np.eye(Npar)), JtWdy)

            # update the [idx] elements
            p_try = p + h.squeeze()
            # apply constraints
            p_try = np.minimum(np.maximum(p_min, p_try), p_max)

            # residual error using p_try
            lm_func.set_p(p_try)
            delta_y_try = (y - lm_func.forward(x)).reshape(-1, 1)

            # floating point error; break
            if not all(np.isfinite(delta_y)):
                break

            # Chi-squared error criteria
            X2_try = delta_y_try.T @ (W @ delta_y_try)

            # TODO % Quadratic
            if Update_Type == 2:
                # One step of quadratic line update in the h direction for minimum X2
                alpha = np.divide(JtWdy.T @ h, ((X2_try - X2) / 2 + 2 * JtWdy.T @ h))
                h = alpha * h

                # % update only [idx] elements
                p_try = p + h
                # % apply constraints
                p_try = np.minimum(np.maximum(p_min, p_try), p_max)

                # % residual error using p_try
                delta_y = y - lm_func(x, p_try)
                func_calls = func_calls + 1
                # % Chi-squared error criteria
                X2_try = delta_y.T @ (delta_y * weight)

            rho = (X2 - X2_try) / np.abs(h.T @ (lambda_ * np.diag(np.diag(JtWJ)) @ h + JtWdy))

            cvg_hst.append({'X2/DoF': X2 / DoF, 'p': p, 'lambda': lambda_, 'y_hat': y_hat})

            # it IS significantly better
            if (rho > epsilon_4):
                # % accept p_try
                p = p_try

                # % decrease lambda ==> Gauss-Newton method
                # % Levenberg
                if Update_Type == 1:
                    lambda_ = max(lambda_ / lambda_DN_fac, 1.e-7)
                # % Quadratic
                elif Update_Type == 2:
                    lambda_ = max(lambda_ / (1 + alpha), 1.e-7)
                # % Nielsen
                else:
                    lambda_ = lambda_ * max(1 / 3, 1 - (2 * rho - 1) ** 3)
                    nu = 2

            # it IS NOT better
            else:

                # % increase lambda  ==> gradient descent method
                # % Levenberg
                if Update_Type == 1:
                    lambda_ = min(lambda_ * lambda_UP_fac, 1.e7)
                # % Quadratic
                elif Update_Type == 2:
                    lambda_ = lambda_ + abs((X2_try - X2) / 2 / alpha)
                # % Nielsen
                else:
                    lambda_ = lambda_ * nu
                    nu = 2 * nu

            # update convergence history ... save _reduced_ Chi-square

            if (np.max(np.abs(h.squeeze()) / (np.abs(p) + 1e-12)) < epsilon_2 and iteration > 2):
                print('**** Convergence in Parameters ****')
                break

            iteration = iteration + 1
            # --- End of Main Loop --- #
            # --- convergence achieved, find covariance and confidence intervals


class Gradient_Descent(optim):
    def back(self):
        pass

class Gauss_Newton(optim):
    def back(self):
        pass


def lm(p_init, x, y, lm_func, use_broyden=False):

    # assert len(x) == len(y), 'The length of x must equal the length of y_dat!'

    # number of parameters
    Npar = len(p_init)
    # number of data points
    Npnt = len(y)
    # statistical degrees of freedom
    DoF = Npnt - Npar + 1


    # weights or a scalar weight value ( weight >= 0 )
    weight = 1/(y.T@y)
    # fractional increment of 'p' for numerical derivatives
    dp = -0.001
    # lower bounds for parameter values
    p_min = -1000
    # upper bounds for parameter values
    p_max = 1000

    MaxIter       = 1000        # maximum number of iterations
    epsilon_1     = 1e-6        # convergence tolerance for gradient
    epsilon_2     = 1e-6        # convergence tolerance for parameters
    epsilon_4     = 1e-5        # determines acceptance of a L-M step
    lambda_0      = 1e-2        # initial value of damping paramter, lambda
    lambda_UP_fac = 11          # factor for increasing lambda
    lambda_DN_fac = 9           # factor for decreasing lambda
    Update_Type   = 1           # 1: Levenberg-Marquardt lambda update, 2: Quadratic update, 3: Nielsen's lambda update equations


    # diagonal weights matrix W
    W = np.eye(Npnt) * weight



    # Marquardt: init'l lambda
    if Update_Type == 1:
        lambda_  = lambda_0
    # Quadratic and Nielsen
    else:
        lambda_  = lambda_0 * max(np.diag(JtWJ))
        nu=2


    # initialize convergence history
    cvg_hst = []
    p = p_init

    iteration=0
    dX2 = 0
    # -------- Start Main Loop ----------- #
    while iteration < MaxIter:

        lm_func.set_p(p)
        # evaluate model using parameters 'p'
        y_hat = lm_func.forward(x)

        # TODO
        if use_broyden:
            if iteration % (2 * Npar) == 0 or dX2 > 0:
                # finite difference
                J = lm_func.get_J(x, y_hat)
            else:
                # rank-1 update
                J = lm_Broyden_J(p_old, y_old, J, p, y_hat)
        else:
            J = lm_func.get_J(x, y_hat)

        # residual error between model and data
        delta_y = (y - y_hat).reshape(-1, 1)

        # Chi-squared error criteria
        X2 = delta_y.T @ (W @ delta_y)
        JtWJ = J.T @ (W @ J)
        JtWdy = J.T @ (W @ delta_y)

        if np.abs(JtWdy).max() < epsilon_1:
            print('*** Your Initial Guess is Extremely Close to Optimal ***')
            print('**** Convergence in r.h.s. ("JtWdy")  ****')
            break
        # incremental change in parameters
        # Marquardt
        if Update_Type == 1:
            h = np.linalg.solve((JtWJ + lambda_*np.diag(np.diag(JtWJ)) ), JtWdy)
        # Quadratic and Nielsen
        else:
            h = np.linalg.solve((JtWJ + lambda_*np.eye(Npar) ), JtWdy)

        # update the [idx] elements
        p_try = p + h.squeeze()
        # apply constraints
        p_try = np.minimum(np.maximum(p_min,p_try),p_max)

        # residual error using p_try
        lm_func.set_p(p_try)
        delta_y_try = (y - lm_func.forward(x)).reshape(-1, 1)

        # floating point error; break
        if not all(np.isfinite(delta_y)):
            break

        # Chi-squared error criteria
        X2_try = delta_y_try.T @ (W@delta_y_try)

        #TODO % Quadratic
        if Update_Type == 2:
          # One step of quadratic line update in the h direction for minimum X2
          alpha =  np.divide(JtWdy.T @ h, ( (X2_try - X2)/2 + 2*JtWdy.T@h ))
          h = alpha * h

          # % update only [idx] elements
          p_try = p + h
          # % apply constraints
          p_try = np.minimum(np.maximum(p_min,p_try),p_max)

          # % residual error using p_try
          delta_y = y - lm_func(x,p_try)
          func_calls = func_calls + 1
          # % Chi-squared error criteria
          X2_try = delta_y.T @ ( delta_y * weight )

        rho = (X2-X2_try)/np.abs(h.T@(lambda_ * np.diag(np.diag(JtWJ))@h+JtWdy))

        cvg_hst.append({'X2/DoF':X2/DoF, 'p':p,'lambda':lambda_,'y_hat':y_hat})

        # it IS significantly better
        if ( rho > epsilon_4 ):
            # % accept p_try
            p = p_try

            # % decrease lambda ==> Gauss-Newton method
            # % Levenberg
            if Update_Type == 1:
                lambda_ = max(lambda_/lambda_DN_fac,1.e-7)
            # % Quadratic
            elif Update_Type == 2:
                lambda_ = max( lambda_/(1 + alpha) , 1.e-7 )
            # % Nielsen
            else:
                lambda_ = lambda_*max( 1/3, 1-(2*rho-1)**3 )
                nu = 2

        # it IS NOT better
        else:

            # % increase lambda  ==> gradient descent method
            # % Levenberg
            if Update_Type == 1:
                lambda_ = min(lambda_*lambda_UP_fac,1.e7)
            # % Quadratic
            elif Update_Type == 2:
                lambda_ = lambda_ + abs((X2_try - X2)/2/alpha)
            # % Nielsen
            else:
                lambda_ = lambda_ * nu
                nu = 2*nu

        # update convergence history ... save _reduced_ Chi-square

        if (np.max(np.abs(h.squeeze())/(np.abs(p)+1e-12)) < epsilon_2  and  iteration > 2 ):
            print('**** Convergence in Parameters ****')
            break

        iteration = iteration + 1
        # --- End of Main Loop --- #
        # --- convergence achieved, find covariance and confidence intervals

    #  ---- Error Analysis ----
    #  recompute equal weights for paramter error analysis
    if np.var(weight) == 0:
        weight = DoF/(delta_y.T@delta_y)

    # % reduced Chi-square
    redX2 = X2 / DoF

    # JtWJ,JtWdy,X2,y_hat,J = lm_matx(x,p_old,y_old,-1,J,p,y,weight,dp)

    # standard error of parameters
    covar_p = np.linalg.inv(JtWJ)
    sigma_p = np.sqrt(np.diag(covar_p))
    error_p = sigma_p/p

    # standard error of the fit
    sigma_y = np.zeros((Npnt,1))
    for i in range(Npnt):
        sigma_y[i,0] = J[i,:] @ covar_p @ J[i,:].T

    sigma_y = np.sqrt(sigma_y)

    # parameter correlation matrix
    corr_p = covar_p / [sigma_p@sigma_p.T]

    # coefficient of multiple determination
    R_sq = np.correlate(y, y_hat)
    R_sq = 0

    # convergence history

    print('\nLM fitting results:')
    print(f'function runs is {lm_func.num_iter + 1} times in {iteration + 1} iters')
    for i in range(Npar):
        print('----------------------------- ')
        print('parameter      = p%i' %(i+1))
        print('fitted value   = %0.4f' % p[i])
        print('standard error = %0.2f %%' % error_p[i])

    return p,cvg_hst

#TODO: unify the mainstream optimization algorithm into one class like torch.optim
def gradient_descent(p_init,x,y):
    pass
