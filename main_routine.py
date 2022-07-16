# -----------------------------------------------------------------------------------------------------------------
#          A.3) Field Inversion Optimization for the k-omega Turbulence Model
#                  A.3.2) Main Routine for the Field Inversion Optimizer
# -----------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------
#  Implementation based on the problem defined in the following publication:
# --------------------------------------------------------------------------------------------------------
#     Eric J. Parish, Karthik Duraisamy,
#     A paradigm for data-driven predictive modeling using field inversion and machine learning,
#     Journal of Computational Physics,
#     Volume 305,
#     2016,
#     Pages 758-774,
#     ISSN 0021-9991,
#     https://doi.org/10.1016/j.jcp.2015.11.012.
#     (http://www.sciencedirect.com/science/article/pii/S0021999115007524)
#     Keywords: Data-driven modeling; Machine learning; Closure modeling
# --------------------------------------------------------------------------------------------------------


import  numpy as np
import  matplotlib
import  time
import  datetime
from    copy import deepcopy
from    scipy.interpolate import splrep,splev
import  os
plt     = matplotlib.pyplot
array   = np.array
pi      = np.pi
sin     = np.sin
exp     = np.exp
myprint = print

# ---------------------------------------------
#     DNS data from Jimenez et al.
#     File   : Re950.prof
#     Website: torroja.dmt.upm.es/channels/data/statistics/Re950/profiles/Re950.prof
# ---------------------------------------------

# NOTE: Please overwrite the variable "Filename_DNS_data" if the data is located elsewhere
#       (or if the Python code is executed interactively and the variable __file__ is not defined)
Filename_DNS_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Re950.prof')



# Matlab tic-toc
def tic():
    global GlobalTimeTic
    GlobalTimeTic=time.time()
def toc(BoolPrint=1):
    vToc=time.time()-GlobalTimeTic
    if BoolPrint: myprint( "Elapsed Time: %.0f s   ( %.3f h ) " % (vToc,vToc/3600.) )
    return vToc
tic()

class Class_Mesh (): pass
class Class_Model(): pass
class Class_DNS  (): pass

###  ---------------------------------------------
# Read Jimenez DNS Data
# File   : Re950.prof
# Website: torroja.dmt.upm.es/channels/data/statistics/Re950/profiles/Re950.prof
###  ---------------------------------------------

def f_read_DNS_data(Filename_DNS_data):
    with open(Filename_DNS_data,'r') as f_read:
        DNS_Text=[i.strip('\n').strip('\r') for i in f_read.readlines()]
    i_Header= [i for i in range(len(DNS_Text)) if '      y/h             y+              U+   ' in DNS_Text[i]]
    if len(i_Header)>1: raise Exception('Multiple Headers Found', i_Header)
    f_weed_void= lambda x: [i for i in x if i!='']
    i_Header=i_Header[0]
    v_Headers= f_weed_void(DNS_Text[i_Header].strip('%').split(' '))
    Dict_Headers=dict(zip(v_Headers, range(len(v_Headers)) ))
    DNS_array = [f_weed_void(DNS_Text[i].split(' ')) for i in range(len(DNS_Text)) if ( (i>i_Header) and not ('%' in DNS_Text[i]) )]
    DNS_array = np.array([list(map(float,i)) for i in DNS_array])
    y_Plus_DNS=DNS_array[:,Dict_Headers['y+']]
    U_Plus_DNS=DNS_array[:,Dict_Headers['U+']]
    
    ReT = [i for i in DNS_Text if ('Re_{\tau}' in i) or (r'Re_{\tau}' in i)]
    if len(ReT)>1: raise Exception('Multiple Re_tau Entries Found', ReT)
    ReT = float(ReT[0].split('tau}')[1].split('=')[1])
    nu = 1./ReT
    vDNSdata=Class_DNS()
    vDNSdata.y_Plus   = y_Plus_DNS
    vDNSdata.U_Plus   = U_Plus_DNS
    vDNSdata.y_coords = y_Plus_DNS*nu
    vDNSdata.nu       = nu
    vDNSdata.ReT      = ReT
    return vDNSdata

DNSdata = f_read_DNS_data(Filename_DNS_data)

# Data Points from Parish & Duraisamy (2016)
# -------------------------------------------------------------------
# # # # Source:
# Eric J. Parish, Karthik Duraisamy,
# A paradigm for data-driven predictive modeling using field inversion and machine learning,
# Journal of Computational Physics,
# Volume 305,
# 2016,
# Pages 758-774,
# ISSN 0021-9991,
# https://doi.org/10.1016/j.jcp.2015.11.012.
# (http://www.sciencedirect.com/science/article/pii/S0021999115007524)
# Keywords: Data-driven modeling; Machine learning; Closure modeling
# -------------------------------------------------------------------
#
YplusBeta_Parish=array([[4.93197382e-02, 9.99989846e-01],
       [1.86249468e+00, 1.00076551e+00], [2.20396226e+00, 1.00313332e+00], [2.45046687e+00, 1.00741988e+00], [2.57982455e+00, 1.01119614e+00],
       [2.71353911e+00, 1.01666661e+00], [2.85144245e+00, 1.02444365e+00], [2.99393005e+00, 1.03534376e+00], [3.14099469e+00, 1.05040796e+00],
       [3.29295051e+00, 1.07086097e+00], [3.44981381e+00, 1.09805001e+00], [3.61195635e+00, 1.13344474e+00], [3.77923369e+00, 1.17853527e+00],
       [3.95205842e+00, 1.23450550e+00], [4.13048755e+00, 1.30202902e+00], [4.31457112e+00, 1.38086090e+00], [4.50480748e+00, 1.46953145e+00],
       [5.11387039e+00, 1.75513882e+00], [5.33017432e+00, 1.83586903e+00], [5.55366057e+00, 1.89641158e+00], [5.78446874e+00, 1.92925480e+00],
       [6.02273630e+00, 1.92884655e+00], [6.26859834e+00, 1.89204337e+00], [6.52284711e+00, 1.81868196e+00], [6.78534833e+00, 1.71184456e+00],
       [7.05627159e+00, 1.57759359e+00], [7.33615654e+00, 1.42360403e+00], [8.23169855e+00, 9.34977021e-01], [8.54998533e+00, 7.92704057e-01],
       [8.87833325e+00, 6.72925326e-01], [9.21789190e+00, 5.80376456e-01], [9.56801704e+00, 5.17261965e-01], [9.92993409e+00, 4.84153393e-01],
       [1.03039772e+01, 4.79519826e-01], [1.06899467e+01, 4.99217589e-01], [1.10881302e+01, 5.38388582e-01], [1.14999820e+01, 5.91031620e-01],
       [1.28166443e+01, 7.68107368e-01], [1.32846330e+01, 8.16953739e-01], [1.37683169e+01, 8.56083907e-01], [1.42674462e+01, 8.83864897e-01],
       [1.47824265e+01, 8.99745581e-01], [1.53152202e+01, 9.05318109e-01], [1.58648095e+01, 9.02113395e-01], [1.64324583e+01, 8.92458429e-01],
       [1.70186960e+01, 8.78374018e-01], [1.88944343e+01, 8.29935891e-01], [1.95605875e+01, 8.18280531e-01], [2.02492026e+01, 8.11054616e-01],
       [2.09599393e+01, 8.08992984e-01], [2.16934278e+01, 8.12524293e-01], [2.24514489e+01, 8.21791428e-01], [2.32347818e+01, 8.36733151e-01],
       [2.40430127e+01, 8.57165753e-01], [2.48780999e+01, 8.82742227e-01], [2.57395880e+01, 9.12931856e-01], [2.66295611e+01, 9.47142686e-01],
       [2.84985615e+01, 1.02501519e+00], [3.04910254e+01, 1.11082804e+00], [3.48824192e+01, 1.28578092e+00], [3.72985607e+01, 1.36632742e+00],
       [3.98760054e+01, 1.43793338e+00], [4.12276405e+01, 1.46973557e+00], [4.26229345e+01, 1.49857800e+00], [4.40654504e+01, 1.52429736e+00],
       [4.55544820e+01, 1.54683242e+00], [4.70914479e+01, 1.56606070e+00], [4.86778072e+01, 1.58202303e+00], [5.03176059e+01, 1.59457653e+00],
       [5.20126441e+01, 1.60384366e+00], [5.37593439e+01, 1.60982443e+00], [5.55647016e+01, 1.61260049e+00], [5.74306873e+01, 1.61221266e+00],
       [5.93563344e+01, 1.60882424e+00], [6.13434454e+01, 1.60257811e+00], [6.33970801e+01, 1.59351509e+00], [6.55161517e+01, 1.58190056e+00],
       [6.77060540e+01, 1.56783656e+00], [6.99691546e+01, 1.55154763e+00], [7.23042428e+01, 1.53315625e+00], [7.72029967e+01, 1.49112755e+00],
       [8.24294811e+01, 1.44346509e+00], [9.09285483e+01, 1.36528640e+00], [1.07045557e+02, 1.23252551e+00], [1.14257639e+02, 1.18333214e+00],
       [1.21943291e+02, 1.13822120e+00], [1.30139342e+02, 1.09800918e+00], [1.38879242e+02, 1.06314516e+00], [1.48206096e+02, 1.03362914e+00],
       [1.58143323e+02, 1.00913451e+00], [1.68746841e+02, 9.88844790e-01], [1.80052219e+02, 9.71637314e-01], [1.98436129e+02, 9.49020608e-01],
       [2.74237804e+02, 8.77455470e-01], [2.92551465e+02, 8.67433084e-01], [3.02161936e+02, 8.64044661e-01], [3.12072330e+02, 8.61962617e-01],
       [3.22324071e+02, 8.61146130e-01], [3.32895748e+02, 8.61554374e-01], [3.55108636e+02, 8.65208156e-01], [3.91208031e+02, 8.72719842e-01],
       [4.04038984e+02, 8.73515917e-01], [4.17290771e+02, 8.72127888e-01], [4.30977193e+02, 8.67616794e-01], [4.45089992e+02, 8.58900789e-01],
       [4.59688182e+02, 8.44898027e-01], [4.74741152e+02, 8.24526661e-01], [4.90311849e+02, 7.96908968e-01], [5.06367623e+02, 7.61432582e-01],
       [5.22975614e+02, 7.18015854e-01], [5.40100998e+02, 6.67291561e-01], [5.76052495e+02, 5.51268674e-01], [5.94946030e+02, 4.92297857e-01],
       [6.14428160e+02, 4.39022041e-01], [6.34548253e+02, 3.97381173e-01], [6.55327200e+02, 3.73682621e-01], [6.76786575e+02, 3.73458086e-01],
       [6.98913306e+02, 3.99953109e-01], [7.21799953e+02, 4.53208513e-01], [7.45436048e+02, 5.29448042e-01], [7.95055549e+02, 7.19260997e-01],
       [8.21048942e+02, 8.12585530e-01], [8.47935049e+02, 8.92213483e-01], [8.75701570e+02, 9.51919139e-01], [9.04789142e+02, 9.68800020e-01],
       [9.33944873e+02, 9.99989846e-01]])
# # -----------------------------------------------------


f_arg_unique=lambda x: np.unique(x,return_index=True)[1]
def f_build_interp(X_target,X0_data,Y0_data):
    # Build Bezier interpolation curve :)
    vsort=np.argsort(X0_data)
    X_data=X0_data[vsort]
    Y_data=Y0_data[vsort]
    X_data = np.append(X_data,2-np.flipud(X_data))
    Y_data = np.append(Y_data,  np.flipud(Y_data))
    
    vfilter=f_arg_unique(X_data)
    X_data = X_data[vfilter]
    Y_data = Y_data[vfilter]
    
    f_fit = splrep(X_data, Y_data)
    Y_target = splev(X_target, f_fit)
    #
    return Y_target

# -----------------------------------------------------
#           Create Mesh
# -----------------------------------------------------

YplusFirst = 0.045
YplusEnd   = 1./DNSdata.nu
MESH                  = Class_Model()
MESH.Grid             = Class_Mesh()
MESH.Grid.N_points    = N_points = 100
MESH.Grid.y_Plus      = y_Plus   = 10**np.linspace(np.log10(YplusFirst),np.log10(YplusEnd),N_points)
MESH.y        = y_coords = y_Plus*DNSdata.nu
MESH.N_points = N_points

#
# Finite Difference Coefficients:
fCoeff_df_dy_Center = lambda dXN,dXS :  (dXN - dXS)/(dXN*dXS);
fCoeff_df_dy_North  = lambda dXN,dXS :  dXS/(dXN*(dXN + dXS));
fCoeff_df_dy_South  = lambda dXN,dXS :  -dXN/(dXS*(dXN + dXS));
fCoeff_d2f_dy_Center= lambda dXN,dXS :  -2./(dXN*dXS);
fCoeff_d2f_dy_North = lambda dXN,dXS :  2./(dXN*(dXN + dXS));
fCoeff_d2f_dy_South = lambda dXN,dXS :  2./(dXS*(dXN + dXS));
#
MESH.Grid.G1_matrix=np.zeros((N_points,N_points))
MESH.Grid.G2_matrix=np.zeros((N_points,N_points))
MESH.Grid.trapz_int=np.zeros((N_points,N_points))
for ii in range(N_points):
    indC=ii;
    indN=ii+1;
    indS=ii-1;
    if ii==(N_points-1):
        indN=ii-1; # Use point from the south (repeated), since the channel is symmetric
    if ii==0:
        indS=ii+2; # Use one further point to the north (interpolate from interior)
    dYN=abs( MESH.y[indN]- MESH.y[indC] ); # must be "abs" due to the symmetry (at i=n-1)
    dYS=( MESH.y[indC]- MESH.y[indS] );    # must have a sign (at i=1)
    #
    MESH.Grid.G2_matrix[ii,indC]+=fCoeff_d2f_dy_Center(dYN,dYS);
    MESH.Grid.G2_matrix[ii,indN]+=fCoeff_d2f_dy_North (dYN,dYS);
    MESH.Grid.G2_matrix[ii,indS]+=fCoeff_d2f_dy_South (dYN,dYS);
    #
    MESH.Grid.G1_matrix[ii,indC]+=fCoeff_df_dy_Center(dYN,dYS);
    MESH.Grid.G1_matrix[ii,indN]+=fCoeff_df_dy_North (dYN,dYS);
    MESH.Grid.G1_matrix[ii,indS]+=fCoeff_df_dy_South (dYN,dYS);
    #
for ii in range(1,N_points):
    dY = MESH.y[ii]-MESH.y[ii-1]
    MESH.Grid.trapz_int[ii:,ii-1]+=dY/2.
    MESH.Grid.trapz_int[ii:,ii]+=dY/2.
#
# -----------------------------------------------------
#     Mesh Created
# -----------------------------------------------------

MESH.kOmConstants = Class_Model()
MESH.kOmConstants.C_mu     =1.0;
MESH.kOmConstants.alpha_k  =0.09;
MESH.kOmConstants.sigma_k  =0.6;
MESH.kOmConstants.alpha_om =0.0708
MESH.kOmConstants.sigma_om =0.5;
MESH.kOmConstants.gamma    =13./25.
MESH.kOmConstants.sigma_d  =0.#
MESH.kOmConstants.beta_1   = 0.075;

MESH.kOmConstants.underrelaxK  = 0.9;
MESH.kOmConstants.underrelaxOm = 0.9;
MESH.kOmConstants.indsKcond  =[0]
MESH.kOmConstants.indsKiter  =np.setdiff1d(range(len(MESH.y)),MESH.kOmConstants.indsKcond)
MESH.kOmConstants.indsOmcond = MESH.kOmConstants.indsKcond 
MESH.kOmConstants.indsOmiter = MESH.kOmConstants.indsKiter 

MESH.FieldVars       =Class_Model()
MESH.FieldVars.nu    =np.copy( DNSdata.nu     );
MESH.FieldVars.u_DNS =f_build_interp(MESH.y,DNSdata.y_coords,DNSdata.U_Plus)#np.copy( DNSdata.U_Plus ); f_build_interp(X_target,X0_data,Y0_data):
MESH.FieldVars.u     = MESH.y*0.
MESH.FieldVars.k     = MESH.y*0. + 0.1 ;
MESH.FieldVars.Om    = MESH.y*0. + 1. ; 
MESH.FieldVars.BetaY = MESH.y*0. + 1. ; 
MESH.tol_solver      = 1e-11


# Create Solver

f_broadcast = lambda x: np.array(np.matrix(x).transpose())
maxAbs = lambda x: abs(x).max()

f_Mult_Ab          = lambda xA,yb:  np.array(    np.matrix(xA)*(    np.matrix(yb.flatten()).transpose()    )    )[:,0]

def f_solveEq(A,b,x,v_inds_iter,v_inds_BC,omegaUnderRelax=1): 
    b-= f_Mult_Ab(A[:,v_inds_BC],x[v_inds_BC])
    A=A[v_inds_iter,:]; A=A[:,v_inds_iter];
    b=b[v_inds_iter];
    if omegaUnderRelax<1:
        b     += (1-omegaUnderRelax)/omegaUnderRelax*np.diag(A)*x[v_inds_iter];  # b     = b + (1-omega)/omega*diag(A).*x(2:n);
        A = A - np.diag(np.diag(A)) + np.diag(np.diag(A))/omegaUnderRelax        # A(logical(eye(size(A))))=diag(A)/omega;
    x[v_inds_iter]=np.linalg.solve(A,b)
    return x

def f_solve_U(self):
    # Nabla * ( mu_eff* Nabla(u)) -1
    # mu_eff*GradL_U=(1 - y/H)
    mu_eff = self.FieldVars.nu + self.FieldVars.nut # )*1. (rho)
    Grad_U = (1-self.y)/mu_eff
    self.FieldVars.u = self.Grid.trapz_int @ Grad_U
    return self
#
def f_solve_KOm(self):
    # %   k-eq:  0 = beta* nut  *(du/dy)^2 - alpha_k*k*om
    # %              + ddy[(nu+nut*sigma_k /C_mu)dkdy]
    # %   om-eq: 0 = gamma*(du/dy)^2 - alpha_om*om^2
    # %              + ddy[(nu+nut*sigma_om/C_mu)domdy]
    C_mu          = self.kOmConstants.C_mu; 
    sigma_om      = self.kOmConstants.sigma_om; 
    alpha_om      = self.kOmConstants.alpha_om; 
    gamma         = self.kOmConstants.gamma; 
    sigma_d       = self.kOmConstants.sigma_d; 
    beta_1        = self.kOmConstants.beta_1; 
    sigma_k       = self.kOmConstants.sigma_k; 
    alpha_k       = self.kOmConstants.alpha_k; 
    #
    nu =  self.FieldVars.nu
    BetaY         = self.FieldVars.BetaY
    #
    underrelaxK  = self.kOmConstants.underrelaxK  
    underrelaxOm = self.kOmConstants.underrelaxOm 
    
    indsKcond    = self.kOmConstants.indsKcond  
    indsKiter    = self.kOmConstants.indsKiter  
    indsOmcond   = self.kOmConstants.indsOmcond 
    indsOmiter   = self.kOmConstants.indsOmiter 
    #
    wallDist      = self.y
    #
    G1_mat = self.Grid.G1_matrix
    G2_mat = self.Grid.G2_matrix
    
    u  = self.FieldVars.u
    k  = self.FieldVars.k
    Om = self.FieldVars.Om
    dkdy  = G1_mat @ k
    dOmdy = G1_mat @ Om
    dUdy  = G1_mat @ u
    
    # strMag = np.fabs(dUdy)
    nut = C_mu*k/Om
    nut = np.minimum(np.maximum(nut,0.),100.)
    
    # %   om-eq: 0 = gamma*(du/dy)^2 - alpha_om*om^2
    # %              + ddy[(nu+nut*sigma_om/C_mu)domdy]
    nu_eff_om = nu + nut*sigma_om/C_mu
    A_om = f_broadcast(nu_eff_om)*G2_mat + f_broadcast(G1_mat @ nu_eff_om )*G1_mat
    A_om-= np.diag(alpha_om*Om)
    
    gradKgradOm = dkdy*dOmdy
    gradKgradOm[gradKgradOm<0]=0.
    b_om = -gamma*(dUdy**2) - sigma_d/Om* gradKgradOm
    
    Om[0]=60.*nu/beta_1/(wallDist[1]**2)
    Om = f_solveEq(A_om,b_om,Om,indsOmiter,indsOmcond,underrelaxOm);
    
    # %   k-eq:  0 = beta* nut  *(du/dy)^2 - alpha_k*k*om
    # %              + ddy[(nu+nut*sigma_k /C_mu)dkdy]
    nu_eff_k = nu + nut*sigma_k/C_mu
    A_k = f_broadcast(nu_eff_k)*G2_mat + f_broadcast(G1_mat @ nu_eff_k )*G1_mat
    A_k-= np.diag(alpha_k*Om)
    
    b_k=-BetaY*nut*(dUdy**2)
    
    k[0]= 0.
    k = f_solveEq(A_k,b_k,k,indsKiter,indsKcond,underrelaxK);
    
    nut = C_mu*k/Om
    self.FieldVars.k  = k 
    self.FieldVars.Om = Om
    self.FieldVars.nut= nut
    return self
#
def f_CFDsolve(self,BoolPrintProgress=0):
    n=self.N_points;
    nmax  = 40000;   nmin=5;
    tol=self.tol_solver
    # tol  = 1e-10;   # iteration limits
    nResid= 25;     # steps to print the residual
    residual = 1e20; iter = 0; 
    
    while ( (residual > tol) and (iter<nmax) ) or (iter<=nmin) : 
        u_old=np.copy(self.FieldVars.u)
        self=f_solve_KOm(self);
        self=f_solve_U (self);
        residual = np.linalg.norm(self.FieldVars.u-u_old);
        if ( (iter % nResid) == 0) and BoolPrintProgress:
            myprint('%d    %12.6e' % (iter, residual) );
        iter += 1;
    if BoolPrintProgress:
        myprint('%d    %12.6e' % (iter, residual) );
    self.Niters_CFDsolve=iter
    return self

MESH=f_CFDsolve(MESH,1)

print('Solver Validated!')

MESH.Optim=Class_Model()
MESH.Optim.DJDB=MESH.FieldVars.BetaY*0.
MESH.Optim.IndsWcond=[0,N_points,2*N_points]
MESH.Optim.IndsWkeep=np.setdiff1d(range(3*N_points),MESH.Optim.IndsWcond)

def f_build_DisAadjMet(self):
    N_points= self.N_points
    
    self.Optim.dRdB = np.zeros((3*N_points,N_points))
    self.Optim.dRdW = np.zeros((3*N_points,3*N_points))
    
    C_mu          = self.kOmConstants.C_mu; 
    sigma_om      = self.kOmConstants.sigma_om; 
    alpha_om      = self.kOmConstants.alpha_om; 
    gamma         = self.kOmConstants.gamma; 
    sigma_d       = self.kOmConstants.sigma_d; 
    sigma_k       = self.kOmConstants.sigma_k; 
    alpha_k       = self.kOmConstants.alpha_k; 
    nu_lam        = self.FieldVars.nu
    
    arr_DISCR_d2nutdy2 = self.Grid.G2_matrix @ self.FieldVars.nut
    arr_DISCR_dnutdy   = self.Grid.G1_matrix @ self.FieldVars.nut
    arr_DISCR_d2omdy2  = self.Grid.G2_matrix @ self.FieldVars.Om
    arr_DISCR_domdy    = self.Grid.G1_matrix @ self.FieldVars.Om
    arr_DISCR_d2kdy2   = self.Grid.G2_matrix @ self.FieldVars.k
    arr_DISCR_dkdy     = self.Grid.G1_matrix @ self.FieldVars.k
    arr_DISCR_d2udy2   = self.Grid.G2_matrix @ self.FieldVars.u
    arr_DISCR_dudy     = self.Grid.G1_matrix @ self.FieldVars.u
    
    ii_last = N_points-1
    for ii in range(1,N_points):
        ind_C= ii
        ind_S= ii-1
        ind_N= ind_S if ii==ii_last else (ii+1)
        u_C     = self.FieldVars.u[ind_C]
        u_N     = self.FieldVars.u[ind_N]
        u_S     = self.FieldVars.u[ind_S]
        k_C     = self.FieldVars.k[ind_C]
        k_N     = self.FieldVars.k[ind_N]
        k_S     = self.FieldVars.k[ind_S]
        omega_C = self.FieldVars.Om[ind_C]
        omega_N = self.FieldVars.Om[ind_N]
        omega_S = self.FieldVars.Om[ind_S]
        cCd2    = self.Grid.G2_matrix[ii,ind_C]
        cNd2    = self.Grid.G2_matrix[ii,ind_N] if ii!=ii_last else 0.
        cSd2    = self.Grid.G2_matrix[ii,ind_S]
        cCd1    = self.Grid.G1_matrix[ii,ind_C]
        cNd1    = self.Grid.G1_matrix[ii,ind_N] if ii!=ii_last else 0.
        cSd1    = self.Grid.G1_matrix[ii,ind_S]
        beta_C  = self.FieldVars.BetaY[ind_C]
        vnutC   = self.FieldVars.nut  [ind_C]
        vDISCR_d2nutdy2 = arr_DISCR_d2nutdy2 [ind_C]
        vDISCR_dnutdy   = arr_DISCR_dnutdy   [ind_C]
        vDISCR_d2omdy2  = arr_DISCR_d2omdy2  [ind_C]
        vDISCR_domdy    = arr_DISCR_domdy    [ind_C]
        vDISCR_d2kdy2   = arr_DISCR_d2kdy2   [ind_C]
        vDISCR_dkdy     = arr_DISCR_dkdy     [ind_C]
        vDISCR_d2udy2   = arr_DISCR_d2udy2   [ind_C]
        vDISCR_dudy     = arr_DISCR_dudy     [ind_C]
        
        vdRn_wi = np.array([[(cSd1* (vDISCR_dnutdy) + cSd2* (vnutC + nu_lam)), (cCd1* (vDISCR_dnutdy) + cCd2* (vnutC + nu_lam)), (cNd1* (vDISCR_dnutdy) + cNd2* (vnutC + nu_lam)), (C_mu* cSd1* (vDISCR_dudy)/ omega_S), (C_mu* cCd1* (vDISCR_dudy)/ omega_C + C_mu* (vDISCR_d2udy2)/ omega_C), (C_mu* cNd1* (vDISCR_dudy)/ omega_N), (-C_mu* cSd1* k_S* (vDISCR_dudy)/ omega_S**2), (-C_mu* cCd1* k_C* (vDISCR_dudy)/ omega_C**2 - C_mu* k_C* (vDISCR_d2udy2)/ omega_C**2), (-C_mu* cNd1* k_N* (vDISCR_dudy)/ omega_N**2), ], [(2* C_mu* beta_C* cSd1* k_C* (vDISCR_dudy)/ omega_C), (2* C_mu* beta_C* cCd1* k_C* (vDISCR_dudy)/ omega_C), (2* C_mu* beta_C* cNd1* k_C* (vDISCR_dudy)/ omega_C), (cSd1* sigma_k* (vDISCR_dkdy)/ omega_S + cSd2* (k_C* sigma_k/ omega_C + nu_lam) + cSd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (C_mu* beta_C* (vDISCR_dudy)**2/ omega_C - alpha_k* omega_C + cCd1* sigma_k* (vDISCR_dkdy)/ omega_C + cCd2* (k_C* sigma_k/ omega_C + nu_lam) + sigma_k* (vDISCR_d2kdy2)/ omega_C + cCd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (cNd1* sigma_k* (vDISCR_dkdy)/ omega_N + cNd2* (k_C* sigma_k/ omega_C + nu_lam) + cNd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (-cSd1* k_S* sigma_k* (vDISCR_dkdy)/ omega_S**2), (-C_mu* beta_C* k_C* (vDISCR_dudy)**2/ omega_C**2 - alpha_k* k_C - cCd1* k_C* sigma_k* (vDISCR_dkdy)/ omega_C**2 - k_C* sigma_k* (vDISCR_d2kdy2)/ omega_C**2), (-cNd1* k_N* sigma_k* (vDISCR_dkdy)/ omega_N**2), ], [(2* cSd1* gamma* (vDISCR_dudy)), (2* cCd1* gamma* (vDISCR_dudy)), (2* cNd1* gamma* (vDISCR_dudy)), (cSd1* sigma_om* (vDISCR_domdy)/ omega_S + cSd1* sigma_d* (vDISCR_domdy)/ omega_C), (cCd1* sigma_d* (vDISCR_domdy)/ omega_C + cCd1* sigma_om* (vDISCR_domdy)/ omega_C + sigma_om* (vDISCR_d2omdy2)/ omega_C), (cNd1* sigma_om* (vDISCR_domdy)/ omega_N + cNd1* sigma_d* (vDISCR_domdy)/ omega_C), (-cSd1* k_S* sigma_om* (vDISCR_domdy)/ omega_S**2 + cSd1* sigma_d* (vDISCR_dkdy)/ omega_C + cSd2* (k_C* sigma_om/ omega_C + nu_lam) + cSd1* sigma_om* (vDISCR_dnutdy)/ C_mu), (-2* alpha_om* omega_C - cCd1* k_C* sigma_om* (vDISCR_domdy)/ omega_C**2 + cCd1* sigma_d* (vDISCR_dkdy)/ omega_C + cCd2* (k_C* sigma_om/ omega_C + nu_lam) - k_C* sigma_om* (vDISCR_d2omdy2)/ omega_C**2 - sigma_d* (vDISCR_dkdy)* (vDISCR_domdy)/ omega_C**2 + cCd1* sigma_om* (vDISCR_dnutdy)/ C_mu), (-cNd1* k_N* sigma_om* (vDISCR_domdy)/ omega_N**2 + cNd1* sigma_d* (vDISCR_dkdy)/ omega_C + cNd2* (k_C* sigma_om/ omega_C + nu_lam) + cNd1* sigma_om* (vDISCR_dnutdy)/ C_mu), ], ]);
        vdRn_bi = np.array([[(0),], [(C_mu*k_C*(vDISCR_dudy)**2/omega_C),], [(0),],]);
        rows_map=np.array([ii])
        inds_map=np.array([ind_S,ind_C,ind_N])
        rows_map=np.append(np.append(rows_map, rows_map+N_points), rows_map+2.*N_points).astype(int)
        inds_map=np.append(np.append(inds_map, inds_map+N_points), inds_map+2.*N_points).astype(int)
        for i in range(3):
            for j in range(9):
                self.Optim.dRdW[rows_map[i],inds_map[j]]+=vdRn_wi[i,j]
            self.Optim.dRdB[rows_map[i],ii]+=vdRn_bi[i,0]
    self.Optim.dJdW = np.zeros(3*N_points)
    self.Optim.dJdW[:N_points]=2*(self.FieldVars.u-self.FieldVars.u_DNS)
    self.Optim.dJdB= 0.
    return self

def f_get_Jacobian(self):
    self=f_CFDsolve        (self,0)
    self=f_build_DisAadjMet(self)
    
    IndsWkeep = self.Optim.IndsWkeep
    dRdW=self.Optim.dRdW
    dRdB=self.Optim.dRdB
    
    dJdW=self.Optim.dJdW
    dJdB=self.Optim.dJdB
    
    dRdB=dRdB[IndsWkeep,:] ; dRdB=dRdB[:,1:]
    dRdW=dRdW[IndsWkeep,:] ; dRdW=dRdW[:,IndsWkeep]
    
    dJdW=dJdW[IndsWkeep] 
    
    Psi=np.linalg.solve(dRdW.transpose(),-dJdW)
    DJDB = Psi @ dRdB   + dJdB 
    self.Optim.DJDB[1:] = DJDB
    self.Optim.Jcost    = np.sum((self.FieldVars.u-self.FieldVars.u_DNS)**2)
    return self

MESH= f_get_Jacobian(MESH)
# plt.plot(MESH.Optim.DJDB)
# Bold Drive Method with Added Momentum
kPlus  = 1.2
kMinus = 0.5
c_1    = 0.9
m_n    = np.zeros(MESH.Optim.DJDB.shape)
alpha  = 0.1 / np.fabs(MESH.Optim.DJDB).max()

MESH.Ref_Betas                 = Class_Model()
MESH.Ref_Betas.float_formatter = lambda x: "%.20e" % x


MESH.Ref_Betas.BetaYFinal_Former= np.array([1.00000000000000000000e+00,    1.00000000000000000000e+00,
    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,
    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,
    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,    1.00000000000000000000e+00,
    1.00000000000000000000e+00,    9.99999999999999000799e-01,    9.99999999999900412995e-01,    9.99999999999477751089e-01,
    9.99999999998402389068e-01,    9.99999999995481170245e-01,    9.99999999987771670540e-01,    9.99999999967868147266e-01,
    9.99999999916960979895e-01,    9.99999999789069282663e-01,    9.99999999473851652887e-01,    9.99999998713570370512e-01,
    9.99999996926072887149e-01,    9.99999992854866737169e-01,    9.99999983964405148029e-01,    9.99999965696333759091e-01,
    9.99999931739310787826e-01,    9.99999880470572644064e-01,    9.99999846315008933395e-01,    1.00000001744106215007e+00,
    1.00000112813359054442e+00,    1.00000567509086213391e+00,    1.00002149114692295839e+00,    1.00007177916658829275e+00,
    1.00022208170786286452e+00,    1.00064945813862604673e+00,    1.00181034968439997002e+00,    1.00481957994648674060e+00,
    1.01222227292733779613e+00,    1.02932485157628494932e+00,    1.06582547989120013376e+00,    1.13602558313960422254e+00,
    1.25323014253253028905e+00,    1.41311267974014453941e+00,    1.57067099404560539533e+00,    1.63872435372018476762e+00,
    1.53865140924188481542e+00,    1.27381899822109656562e+00,    9.44272273181597232572e-01,    6.82431956893036928413e-01,
    5.72934304661665394498e-01,    6.14119949348205684814e-01,    7.33209704904382753021e-01,    8.40129072704467017019e-01,
    8.82892644120351288706e-01,    8.64308247313457900241e-01,    8.21135504431931395608e-01,    7.94666589535528600230e-01,
    8.11632961216389836601e-01,    8.78641299151327048733e-01,    9.86273666427910922216e-01,    1.11717095579077185796e+00,
    1.25312389096782927922e+00,    1.37890696367545695367e+00,    1.48346418923621947705e+00,    1.55978108109781476998e+00,
    1.60436549319607135722e+00,    1.61670964664977456771e+00,    1.59878653723584540636e+00,    1.55457322225655270032e+00,
    1.48967829440887400416e+00,    1.41095280498090436261e+00,    1.32592845985855500857e+00,    1.24216443957985145374e+00,
    1.16630443953191620388e+00,    1.10303332798444220231e+00,    1.05455078400233470681e+00,    1.02023548768708360868e+00,
    9.96597354409062052838e-01,    9.77960745153325761159e-01,    9.58141954184070110401e-01,    9.33132116665294497615e-01,
    9.02525986952313563627e-01,    8.70428428558836175810e-01,    8.45552402526347113287e-01,    8.35412200334732291118e-01,
    8.41008681102708988142e-01,    8.53601322143027196887e-01,    8.49910810991996279462e-01,    7.92945233570415908275e-01,
    6.51682713463897500539e-01,    4.46060074499367797962e-01,    3.16852329457211256969e-01,    4.53915025614118783359e-01,
    7.90252239852732185632e-01,    1.00000000000000000000e+00])

MESH.Ref_Betas.BetaParish_Orig = f_build_interp(MESH.y,YplusBeta_Parish[:,0]* MESH.FieldVars.nu, YplusBeta_Parish[:,1])

MESH.Ref_Betas.BetaSecondMin_Worked= np.array([9.99921178972190460854e-01,
    9.99996696905905757546e-01,    1.00007923613135152330e+00,    1.00016933125418239214e+00,    1.00026752921643957706e+00,
    1.00037438145584323124e+00,    1.00049043343924703997e+00,    1.00061621097600483843e+00,    1.00075220261293629243e+00,
    1.00089883729854345340e+00,    1.00105645638293783151e+00,    1.00122527889684254099e+00,    1.00140535893672155687e+00,
    1.00159653388767422300e+00,    1.00179836216278639327e+00,    1.00201004915953317820e+00,    1.00223036027832601569e+00,
    1.00245752018511313786e+00,    1.00268909812768192147e+00,    1.00292188017778216569e+00,    1.00315173096706389266e+00,
    1.00337345009769696169e+00,    1.00358063233109606749e+00,    1.00376554644194793120e+00,    1.00391905602519226903e+00,
    1.00403061761614909031e+00,    1.00408840866639348377e+00,    1.00407966218544308923e+00,    1.00399131889183723665e+00,
    1.00381115512708651849e+00,    1.00352961037819521373e+00,    1.00314262830633782464e+00,    1.00265594768549659577e+00,
    1.00209144444900366544e+00,    1.00149634473112580935e+00,    1.00095641465650753155e+00,    1.00061460134762092444e+00,
    1.00069707282714448482e+00,    1.00154921748764724754e+00,    1.00368807630879719461e+00,    1.00832824889219607734e+00,
    1.01845226571757652145e+00,    1.03962079073003588192e+00,    1.08185717450504403914e+00,    1.15994310019833624104e+00,
    1.29018763400864289181e+00,    1.47632379727627416344e+00,    1.68417360179347963545e+00,    1.82726520979930251443e+00,
    1.79899985579087928755e+00,    1.55614263292952981210e+00,    1.17103404385366194340e+00,    7.91221740339477008774e-01,
    5.58466582882908513241e-01,    5.30468979662427009281e-01,    6.56802826073821788277e-01,    8.15873915305869057413e-01,
    9.07552724502361463088e-01,    9.05955129620918730993e-01,    8.47909008267437758199e-01,    7.86868726926705619462e-01,
    7.63712437344371330994e-01,    7.94099919879319604554e-01,    8.73984300337309782947e-01,    9.90120468860779645581e-01,
    1.12514672685111905004e+00,    1.26217747955314751884e+00,    1.38743715432280567690e+00,    1.49040669902597033847e+00,
    1.56484391316037485531e+00,    1.60774435568543139219e+00,    1.61879260063217333965e+00,    1.60002384373409078933e+00,
    1.55543221614227378247e+00,    1.49050698720611718429e+00,    1.41202835597652498478e+00,    1.32724178299619466337e+00,
    1.24359963721695243599e+00,    1.16751505938493616021e+00,    1.10365680591107584441e+00,    1.05418080121671553506e+00,
    1.01856536592182145284e+00,    9.93567864921946819479e-01,    9.73703936861988861295e-01,    9.52838443215049402113e-01,
    9.27192427609206504258e-01,    8.98126280729280912496e-01,    8.69402407923599240824e-01,    8.48732767611185323631e-01,
    8.43829029650253525929e-01,    8.53734887738358705356e-01,    8.66761408136853095385e-01,    8.55975652340188131184e-01,
    7.82697323845186643254e-01,    6.21976754296574063652e-01,    4.17180149563377900002e-01,    3.32452683951485805647e-01,
    5.22994482057492882099e-01,    8.44630480914004611037e-01,    9.99990079905520357073e-01,])

MESH.FieldVars.BetaY = np.copy(np.ones(MESH.y.shape))

# NOTE:
#     To restore the previous progress, please use the following line of code:
# MESH.FieldVars.BetaY = np.copy(MESH.Ref_Betas.BetaYFinal_Former)

BoolOptimize=True
iiter   =0

tic() # N_iters_print = 100
vkey = True if BoolOptimize else False
while vkey:
    iiter  +=1
    BackupModel = deepcopy(MESH)
    m_n = (c_1*m_n + (1-c_1)*MESH.Optim.DJDB)
    MESH.FieldVars.BetaY -= alpha*m_n
    #
    MESH=f_get_Jacobian(MESH)
    if MESH.Optim.Jcost<BackupModel.Optim.Jcost:
        alpha*=kPlus
    else:
        alpha *=kMinus
        MESH   = deepcopy(BackupModel)
        m_n    = np.copy(MESH.Optim.DJDB)
    if alpha<1e-14: vkey=False
    if toc(0)>=60: print('Iteration %8d: Jcost=%12.4e (alpha = %12.4e)' % (iiter,MESH.Optim.Jcost,alpha));tic()
print('Iteration END: Jcost=%12.4e (alpha = %12.4e)' % (MESH.Optim.Jcost,alpha))

def f_get_Pk(self):
    self=f_get_Jacobian(self)
    dudy = self.Grid.G1_matrix @ self.FieldVars.u
    nut = self.FieldVars.nut
    BetaY = self.FieldVars.BetaY
    Mod_Pk = BetaY*nut*(dudy**2)
    self.FieldVars.Mod_Pk=np.copy(Mod_Pk)
    self.yPlus=self.Grid.y_Plus
    return self

MESH_Beta1 = deepcopy(MESH)
MESH_Beta1.FieldVars.BetaY = MESH_Beta1.y*0.+1.
MESH_Beta1=f_get_Pk(MESH_Beta1)

MESH_ParishOrig = deepcopy(MESH)
MESH_ParishOrig.FieldVars.BetaY = MESH_ParishOrig.Ref_Betas.BetaParish_Orig +0.
MESH_ParishOrig=f_get_Pk(MESH_ParishOrig)

MESH_SecondMinWorked = deepcopy(MESH)
MESH_SecondMinWorked.FieldVars.BetaY = MESH_SecondMinWorked.Ref_Betas.BetaSecondMin_Worked +0.
MESH_SecondMinWorked=f_get_Pk(MESH_SecondMinWorked)

MESH_Former = deepcopy(MESH)
MESH_Former.FieldVars.BetaY = MESH_Former.Ref_Betas.BetaYFinal_Former +0.
MESH_Former=f_get_Pk(MESH_Former)

N_total=3
fig, axarr = plt.subplots(N_total,1) 
ax1,ax2,ax3=axarr

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': "Times New Roman"})

self=MESH_Beta1       ;ax1.semilogx(self.yPlus, self.FieldVars.u, 'y--', linewidth=5, label='Standard')
self=MESH_Former      ;ax1.semilogx(self.yPlus,self.FieldVars.u,'b',linewidth=5,label='First Local Min.')
self=MESH_SecondMinWorked;ax1.semilogx(self.yPlus, self.FieldVars.u, 'r--', linewidth=5, label='Second Local Min.')
self=MESH_Beta1       ;ax1.semilogx(self.yPlus[::3], self.FieldVars.u_DNS[::3], 'go', markersize=10, label='DNS Data')
ax1.legend(loc='best')
ax1.set_ylabel(r'$U^+$')
ax1.grid()

# self=MESH_Beta1       ;ax2.semilogx(self.yPlus, self.FieldVars.BetaY, 'y--', linewidth=5, label='Standard')
self=MESH_Former      ;ax2.semilogx(self.yPlus, self.FieldVars.BetaY, 'b', linewidth=5, label='First Local Min.')
self=MESH_SecondMinWorked;ax2.semilogx(self.yPlus, self.FieldVars.BetaY, 'r', linewidth=5, label='Second Local Min.')
self=MESH_ParishOrig  ;ax2.semilogx(self.yPlus[::], self.FieldVars.BetaY[::],  'g--', linewidth=5, label='Parish et al. (2016)')
ax2.legend(loc='best')
ax2.set_ylabel(r'$\beta(y)$')
ax2.grid()

self=MESH_Beta1       ;ax3.semilogx(self.yPlus, self.FieldVars.Mod_Pk, 'y--', linewidth=5, label='Standard')
self=MESH_Former      ;ax3.semilogx(self.yPlus, self.FieldVars.Mod_Pk, 'b', linewidth=5, label='First Local Min.')
self=MESH_SecondMinWorked;ax3.semilogx(self.yPlus, self.FieldVars.Mod_Pk, 'r', linewidth=5, label= 'Second Local Min.')
ax3.legend(loc='best')
ax3.set_ylabel(r'$\beta\cdot P_k$')
ax3.set_xlabel(r'$Y^+$')
ax3.grid()

fig.set_size_inches(fig.get_size_inches()*1.1*np.array([1.6,3.2]))

plt.savefig("kOmOptim_Parish2015.pdf", bbox_inches = 'tight')
plt.show()