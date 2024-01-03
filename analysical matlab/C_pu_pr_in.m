% 磁标势对r的偏导数（当r<a时）
function pu_pr_in=C_pu_pr_in(a,Br,r,Legendre_Polynomials_L0,Legendre_costh,Num_Legendre)
c_sum = 0;
mu0 = 4*pi*10^(-7);
for k = 2:floor(Num_Legendre/2)   % 注意求和起点为k==2，floor为向下取整函数
    c_sum = c_sum + (1/(2*k-3)+1/(2*k)-(2*k-2)/(2*k-3)*(r/a)^(2*k-3))*Legendre_Polynomials_L0(2*k-2+1)*Legendre_costh(2*k-2+1);
end
pu_pr_in = a*Br/(2*mu0)*(-1/(2*a)+1/a*c_sum);