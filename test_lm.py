# Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for
# nonlinear least squares curve-fitting problems.
# https://people.duke.edu/~hpgavin/ce281/lm.pdf

import numpy as np
from LM import lm
from LM import make_lm_plots

class Func:
    def __init__(self,p):
        self.p = p
        self.num_iter=0

    def forward(self,x):
        y_hat = self.p[0] * np.exp(-x / self.p[1]) + self.p[2] * np.sin(x / self.p[3])
        self.num_iter+=1
        return y_hat

    def set_p(self,p):
        self.p = p

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
    """

    Main function for performing Levenberg-Marquardt curve fitting.

    Parameters
    ----------
    x           : x-values of input data (m x 1), must be 2D array
    y           : y-values of input data (m x 1), must be 2D array
    p_init      : initial guess of parameters values (n x 1), must be 2D array
                  n = 4 in this example

    Returns
    -------
    p       : least-squares optimal estimate of the parameter values
    Chi_sq  : reduced Chi squared error criteria - should be close to 1
    sigma_p : asymptotic standard error of the parameters
    sigma_y : asymptotic standard error of the curve-fit
    corr    : correlation matrix of the parameters
    R_sq    : R-squared cofficient of multiple determination
    cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
              col 3 through n: parameter values). Row number corresponds to
              iteration number.

    """

    # define true fitted parameters for testing (must be 2D array)
    # define initial guess of parameters (must be 2D array)
    # number of data points (x-values will range from 0 to 99)
    Npnt = 100
    # adding noise to input data to simulate artificial measurements
    msmnt_err = 0.05
    p_true = np.array([6, 20, 1, 5])
    x = np.array(range(Npnt))
    lm_func = Func(p_true)
    y_true = lm_func.forward(x)
    # add Gaussian random measurement noise
    y = y_true + msmnt_err * np.random.randn((Npnt))


    p_init = np.array([10, 50, 6, 5.7])
    p_fit, cvg_hst = lm(p_init, x, y, lm_func)

    print(p_fit)
    # plot results of L-M least squares analysis
    # make_lm_plots(x, y, cvg_hst)


if __name__ == '__main__':
    main()

    # flag for making noisy test data
