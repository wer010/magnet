# Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for
# nonlinear least squares curve-fitting problems.
# https://people.duke.edu/~hpgavin/ce281/lm.pdf

import numpy as np
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
# from test_lm import Func

def lm_Broyden_J(p_old,y_old,J,p,y):
    """
    Carry out a rank-1 update to the Jacobian matrix using Broyden's equation.

    Parameters
    ----------
    p_old :     previous set of parameters (n x 1)
    y_old :     model evaluation at previous set of parameters, y_hat(t,p_old) (m x 1)
    J     :     current version of the Jacobian matrix (m x n)
    p     :     current set of parameters (n x 1)
    y     :     model evaluation at current  set of parameters, y_hat(t,p) (m x 1)

    Returns
    -------
    J     :     rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j) (m x n)

    """

    h = p - p_old

    a = (np.array([y - y_old]).T - J@h)@h.T
    b = h.T@h

    # Broyden rank-1 update eq'n
    J = J + a/b

    return J

def lm_matx(x,p_old,y_old,dX2,J,p,y_dat,weight):
    """
    Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, and
    calculate the Chi-squared error function, Chi_sq used by Levenberg-Marquardt
    algorithm (lm).

    Parameters
    ----------
    t      :     independent variables used as arg to lm_func (m x 1)
    p_old  :     previous parameter values (n x 1)
    y_old  :     previous model ... y_old = y_hat(t,p_old) (m x 1)
    dX2    :     previous change in Chi-squared criteria (1 x 1)
    J      :     Jacobian of model, y_hat, with respect to parameters, p (m x n)
    p      :     current parameter values (n x 1)
    y_dat  :     data to be fit by func(t,p,c) (m x 1)
    weight :     the weighting vector for least squares fit inverse of
                 the squared standard measurement errors
    dp     :     fractional increment of 'p' for numerical derivatives
                  - dp(j)>0 central differences calculated
                  - dp(j)<0 one sided differences calculated
                  - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed

    Returns
    -------
    JtWJ   :     linearized Hessian matrix (inverse of covariance matrix) (n x n)
    JtWdy  :     linearized fitting vector (n x m)
    Chi_sq :     Chi-squared criteria: weighted sum of the squared residuals WSSR
    y_hat  :     model evaluated with parameters 'p' (m x 1)
    J :          Jacobian of model, y_hat, with respect to parameters, p (m x n)

    """

    global iteration,func_calls

    W = np.eye(weight.shape[0]) * weight
    # number of parameters
    Npar   = len(p)

    # evaluate model using parameters 'p'
    y_hat = lm_func(x,p)

    func_calls = func_calls + 1

    if iteration%(2*Npar)==0 or dX2 > 0:
        # finite difference
        J = lm_FD_J(x,p,y_hat)
    else:
        # rank-1 update
        J = lm_Broyden_J(p_old,y_old,J,p,y_hat)

    # residual error between model and data
    delta_y = (y_dat - y_hat).reshape(-1,1)

    # Chi-squared error criteria
    error = delta_y.T @ ( W@delta_y )

    JtWJ  = J.T @ (W@J)

    JtWdy = J.T @ ( W@delta_y )


    return JtWJ, JtWdy, error, y_hat, J


def lm(p_init, x, y, lm_func):
    """

    Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals

    Parameters
    ----------
    p : initial guess of parameter values (n x 1)
    t : independent variables (used as arg to lm_func) (m x 1)
    y_dat : data to be fit by func(t,p) (m x 1)

    Returns
    -------
    p       : least-squares optimal estimate of the parameter values
    redX2   : reduced Chi squared error criteria - should be close to 1
    sigma_p : asymptotic standard error of the parameters
    sigma_y : asymptotic standard error of the curve-fit
    corr_p  : correlation matrix of the parameters
    R_sq    : R-squared cofficient of multiple determination
    cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
              col 3 through n: parameter values). Row number corresponds to
              iteration number.

    """

    assert len(x) == len(y), 'The length of x must equal the length of y_dat!'


    # number of parameters
    Npar = len(p_init)
    # number of data points
    Npnt = len(y)
    # previous set of parameters
    p_old  = np.zeros((Npar,1))
    # previous model, y_old = y_hat(t,p_old)
    y_old  = np.zeros((Npnt,1))
    # Jacobian matrix
    J = np.zeros((Npnt,Npar))
    # statistical degrees of freedom
    DoF = Npnt - Npar + 1


    # weights or a scalar weight value ( weight >= 0 )
    weight = 1/(y.T@y)
    # fractional increment of 'p' for numerical derivatives
    dp = -0.001
    # lower bounds for parameter values
    p_min = -100*abs(p_init)
    # upper bounds for parameter values
    p_max = 100*abs(p_init)

    MaxIter       = 1000        # maximum number of iterations
    epsilon_1     = 1e-3        # convergence tolerance for gradient
    epsilon_2     = 1e-3        # convergence tolerance for parameters
    epsilon_4     = 1e-1        # determines acceptance of a L-M step
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
        J = lm_func.get_J(x, y_hat)
        # TODO
        # if iteration % (2 * Npar) == 0 or dX2 > 0:
        #     # finite difference
        #     J = lm_func.get_J(x, y_hat)
        # else:
        #     # rank-1 update
        #     J = lm_Broyden_J(p_old, y_old, J, p, y_hat)

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
        cvg_hst.append({'X2/DoF':X2/DoF, 'p':p})

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
    for i in range(Npar):
        print('----------------------------- ')
        print('parameter      = p%i' %(i+1))
        print('fitted value   = %0.4f' % p[i])
        print('standard error = %0.2f %%' % error_p[i])

    return p,cvg_hst

def make_lm_plots(x,y,cvg_hst):
    # extract parameters data
    p_hst  = cvg_hst[:,2:]
    p_fit  = p_hst[-1,:]
    y_fit = lm_func(x,np.array([p_fit]).T)

    # define fonts used for plotting
    font_axes = {'family': 'serif',
            'weight': 'normal',
            'size': 12}
    font_title = {'family': 'serif',
                  'weight': 'normal',
            'size': 14}

    # define colors and markers used for plotting
    n = len(p_fit)
    colors = pl.cm.ocean(np.linspace(0,.75,n))
    markers = ['o','s','D','v']
    fig = plt.figure(figsize=(12, 4))

    # create plot of raw data and fitted curve
    ax1 = fig.add_subplot(1,3,1)
    ax1.plot(x,y,'wo',markeredgecolor='black',label='Raw data')
    ax1.plot(x,y_fit,'r--',label='Fitted curve',linewidth=2)
    ax1.set_xlabel('t',fontdict=font_axes)
    ax1.set_ylabel('y(t)',fontdict=font_axes)
    ax1.set_title('Data fitting',fontdict=font_title)
    ax1.legend()

    # create plot showing convergence of parameters
    ax2 = fig.add_subplot(1,3,2)
    for i in range(n):
        ax2.plot(cvg_hst[:,0],p_hst[:,i]/p_hst[0,i],color=colors[i],marker=markers[i],
                 linestyle='-',markeredgecolor='black',label='p'+'${_%i}$'%(i+1))
    ax2.set_xlabel('Function calls',fontdict=font_axes)
    ax2.set_ylabel('Values (norm.)',fontdict=font_axes)
    ax2.set_title('Convergence of parameters',fontdict=font_title)
    ax2.legend()

    # create plot showing histogram of residuals
    ax3 = fig.add_subplot(1,3,3)
    sns.histplot(ax=ax3,data=y_fit-y,color='deepskyblue')
    ax3.set_xlabel('Residual error',fontdict=font_axes)
    ax3.set_ylabel('Frequency',fontdict=font_axes)
    ax3.set_title('Histogram of residuals',fontdict=font_title)
    plt.show()
    # create plot showing objective function surface plot
    fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
    # define range of values for gridded parameter search
    p2 = np.arange(0.1*p_fit[1], 2.5*p_fit[1], 0.1)
    p4 = np.arange(0.1*p_fit[3], 2.5*p_fit[3], 0.1)
    X2 = np.zeros((len(p4),len(p2)))
    # gridded parameter search
    for i in range(len(p2)):
        for j in range(len(p4)):
            pt = np.array([[p_hst[-1,0],p2[i],p_hst[-1,2],p4[j]]]).T
            delta_y = y - lm_func(x,pt)
            X2[j,i] = np.log((delta_y.T @ delta_y)/(len(x)-len(p_fit)))
    p2_grid, p4_grid = np.meshgrid(p2, p4)
    # make surface plot
    ax4.plot_surface(p2_grid, p4_grid, X2, cmap='coolwarm', antialiased=True)
    ax4.set_xlabel('P$_2$',fontdict=font_axes)
    ax4.set_ylabel('P$_4$',fontdict=font_axes)
    ax4.set_zlabel('log$_{10}$($\chi$$^2$)',fontdict=font_axes,rotation=90)
    ax4.set_title('Objective Function',fontdict=font_title)
    ax4.zaxis.set_rotate_label(False)
    ax4.azim = 225
    plt.show()