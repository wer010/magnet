# Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for
# nonlinear least squares curve-fitting problems.
# https://people.duke.edu/~hpgavin/ce281/lm.pdf

import numpy as np
from LM import lm
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
class Func:
    # 2D xz plane magnetic dipole model
    def __init__(self,p):
        # p means 2D position [x,z] of sensor array
        # r means 2D rotation [r] of sensor array
        self.set_p(p)
        self.num_iter=0

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


def make_lm_plots(x,y,cvg_hst,lm_func):
    # extract parameters data
    y_fit = []
    p = []
    l = []
    for item in cvg_hst:
        y_fit.append(item['y_hat'])
        p.append(item['p'])
        l.append(item['lambda'])
    y_fit = np.stack(y_fit)
    p = np.stack(p)
    l = np.stack(l)
    # define fonts used for plotting
    font_axes = {'family': 'serif',
            'weight': 'normal',
            'size': 12}
    font_title = {'family': 'serif',
                  'weight': 'normal',
            'size': 14}

    # define colors and markers used for plotting
    n = len(p[0])
    fig = plt.figure(figsize=(12, 4))

    # create plot of raw data and fitted curve
    ax1 = fig.add_subplot(1,3,1)
    x_a = np.arange(len(y))
    ax1.plot(x_a,y,'wo',markeredgecolor='black',label='Raw data')
    ax1.plot(x_a,y_fit[-1],'r--',label='Fitted curve',linewidth=2)
    ax1.set_xlabel('t',fontdict=font_axes)
    ax1.set_ylabel('y(t)',fontdict=font_axes)
    ax1.set_title('Data fitting',fontdict=font_title)
    ax1.legend()

    # create plot showing convergence of parameters
    ax2 = fig.add_subplot(1,3,2)
    colors = pl.cm.ocean(np.linspace(0,.75,n))
    markers = ['o','s','D','v']

    for i in range(n):
        ax2.plot(np.arange(len(p)),p[:,i],color=colors[i],marker=markers[i],
                 linestyle='-',markeredgecolor='black',label='p'+'${_%i}$'%(i+1))
    ax2.plot(np.arange(len(p)),l*10,label = '100*lambda')
    ax2.set_xlabel('Function calls',fontdict=font_axes)
    ax2.set_ylabel('Values (norm.)',fontdict=font_axes)
    ax2.set_title('Convergence of parameters',fontdict=font_title)
    ax2.legend()
    print(l)
    # create plot showing histogram of residuals
    ax3 = fig.add_subplot(1,3,3)
    sns.histplot(ax=ax3,data=y_fit[-1]-y,color='deepskyblue')
    ax3.set_xlabel('Residual error',fontdict=font_axes)
    ax3.set_ylabel('Frequency',fontdict=font_axes)
    ax3.set_title('Histogram of residuals',fontdict=font_title)
    plt.show()
    # create plot showing objective function surface plot
    fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
    # define range of values for gridded parameter search
    p0 = np.arange(min(p[:,0])-5, max(p[:,0])+5, (max(p[:,0])-min(p[:,0]))/200)
    p1 = np.arange(min(p[:,1])-5, max(p[:,1])+5, (max(p[:,1])-min(p[:,1]))/200)
    X2 = np.zeros((len(p1),len(p0)))
    # gridded parameter search
    for i in tqdm(range(len(p0))):
        for j in range(len(p1)):
            pt = np.array([p0[i],p1[j]])
            lm_func.set_p(pt)
            delta_y = y - lm_func.forward(x)
            X2[j,i] = np.log((delta_y.T @ delta_y)/(len(x)-len(p[-1])))
    p0_grid, p1_grid = np.meshgrid(p0, p1)
    # make surface plot
    ax4.plot_surface(p0_grid, p1_grid, X2, cmap='coolwarm', alpha=.6,antialiased=True)
    x_a = p[:,0]
    y_a = p[:,1]
    z_a = []
    for item in p:
        lm_func.set_p(item)
        delta_y = y - lm_func.forward(x)
        z_a.append(np.log((delta_y.T @ delta_y)/(len(x)-len(p[-1]))))
    z_a = np.stack(z_a)
    ax4.plot(x_a,y_a,z_a.flatten(),color = 'black', marker='o')

    ax4.set_xlabel('X',fontdict=font_axes)
    ax4.set_ylabel('Z',fontdict=font_axes)
    ax4.set_zlabel('log$_{10}$($\chi$$^2$)',fontdict=font_axes,rotation=90)
    ax4.set_title('Objective Function',fontdict=font_title)
    ax4.zaxis.set_rotate_label(False)
    ax4.azim = 225
    plt.show()



def main():
    # define true fitted parameters for testing (must be 2D array)
    # define initial guess of parameters (must be 2D array)
    # number of data points (x-values will range from 0 to 99)
    # adding noise to input data to simulate artificial measurements
    msmnt_err = 0.05
    p_true = np.array([200,50,np.pi/4])
    x = np.array([[0,0],
                    [0,50],
                    [50,0],
                    [50,50]])
    lm_func = Func(p_true)
    y_true = lm_func.forward(x)
    # add Gaussian random measurement noise
    y = y_true #+ msmnt_err * np.random.randn((Npnt))


    p_init = np.array([5,10,0])
    p_fit, cvg_hst = lm(p_init, x, y, lm_func)

    print(p_fit)
    # plot results of L-M least squares analysis
    make_lm_plots(x, y, cvg_hst, lm_func)


if __name__ == '__main__':
    main()

    # flag for making noisy test data
