# -----------------------------------------------------------------------------------------------------------------
#          A.3) Field Inversion Optimization for the k-omega Turbulence Model
#                  A.3.1) Sympy Code to Derive the Matrices Required by the Discrete Adjoint Method
# -----------------------------------------------------------------------------------------------------------------

from sympy import init_printing
init_printing()
from IPython.display import display
from sympy import Function,diff,Eq,dsolve,symbols,exp,solve,sin,Matrix,Array, sympify,sqrt,integrate,pi,cos,sin,ln,limit,oo,nsolve
from sympy.tensor import MutableDenseNDimArray
# ***************************************************
# # # Initialize variables employed in the code (mandatory step)
# ***************************************************
mk_symbols = lambda x: symbols(x,real=True)

nu_lam,C_mu,y,alpha_om,sigma_om, alpha_k,sigma_k,gamma,sigma_d                                                        =  mk_symbols('nu_lam,C_mu,y,alpha_om,sigma_om, alpha_k,sigma_k,gamma,sigma_d')
u_C,u_N,u_S,k_C,k_N,k_S,omega_C,omega_N,omega_S,cNd2,cCd2, cSd2,cNd1,cCd1,cSd1,beta_C                                 =  mk_symbols('u_C,u_N,u_S,k_C,k_N,k_S,omega_C,omega_N,omega_S,cNd2,cCd2, cSd2,cNd1,cCd1,cSd1,beta_C')
vDISCR_d2nutdy2,vDISCR_dnutdy,vDISCR_d2omdy2,vDISCR_domdy, vDISCR_d2kdy2,vDISCR_dkdy,vDISCR_d2udy2,vDISCR_dudy,vnutC  =  mk_symbols('vDISCR_d2nutdy2,vDISCR_dnutdy,vDISCR_d2omdy2,vDISCR_domdy, vDISCR_d2kdy2,vDISCR_dkdy,vDISCR_d2udy2,vDISCR_dudy,vnutC')

mk_function = lambda f,x: Function(f)(x)

# ***************************************************
#     Build main variables
# ***************************************************

uC  =  u_C  
uN  =  u_N  
uS  =  u_S 

kC  =  k_C  
kN  =  k_N  
kS  =  k_S  

omC =omega_C 
omN =omega_N 
omS =omega_S 

bC=beta_C

u     =  mk_function('u',y)
k     =  mk_function('k',y)
om    =  mk_function('omega',y)
beta  =  mk_function('beta',y)
nu_T  =  mk_function('nu_T',y)#C_mu*k/om

nutN=C_mu*kN/omN
nutC=C_mu*kC/omC
nutS=C_mu*kS/omS

# Express discretized derivatives as sums of values and coefficients
DISCR_d2udy2    =cNd2* uN+cCd2* uC+cSd2* uS
DISCR_d2kdy2    =cNd2* kN+cCd2* kC+cSd2* kS
DISCR_d2omdy2   =cNd2*omN+cCd2*omC+cSd2*omS

DISCR_dnutdy   = cNd1* nutN+cCd1* nutC+cSd1* nutS
DISCR_d2nutdy2 = cNd2* nutN+cCd2* nutC+cSd2* nutS

DISCR_d2BetaDy2 =0

DISCR_dudy   =cNd1* uN+cCd1* uC+cSd1* uS
DISCR_dkdy   =cNd1* kN+cCd1* kC+cSd1* kS
DISCR_domdy  =cNd1*omN+cCd1*omC+cSd1*omS
DISCR_BetaDy =0

# Build Effective Viscosities beta
nuEffU =nu_lam+nu_T
nueffK =nu_lam+nu_T*sigma_k /C_mu
nueffOm=nu_lam+nu_T*sigma_om/C_mu

# ***************************************************
# # # Build residual eqs.
# ***************************************************
#
#   U-eq:  0 = ddy[(nu+nut)dudy]
R_U =                                                ( nuEffU *(u.diff(y) ) ).diff(y)
#
#   k-eq:  0 =  beta*nut  *(du/dy)^2 - alpha_k*k*om
#              + ddy[(nu+nut*sigma_k /C_mu)dkdy] 
R_k =beta*nu_T *((u.diff(y))**2)  - alpha_k *k *om + ( nueffK *(k.diff(y) ) ).diff(y) 
#
#   om-eq: 0 = gamma*(du/dy)^2 - alpha_om*om^2
#              + ddy[(nu+nut*sigma_om/C_mu)domdy] + dkdy*domdy*sigma_d/om
R_Om=gamma*((u.diff(y))**2)       - alpha_om*om*om + ( nueffOm*(om.diff(y)) ).diff(y) + (k.diff(y)*om.diff(y))*sigma_d/om
#
# ***************************************************
# # # Build discretizations for R_i....
# ***************************************************
def f_discretize(i):
    # Substitute values by discretized approximations
    i=i.subs(u.diff(y).diff(y),DISCR_d2udy2)
    i=i.subs(u.diff(y),DISCR_dudy)
    i=i.subs(u,uC)
    
    i=i.subs(k.diff(y).diff(y),DISCR_d2kdy2)
    i=i.subs(k.diff(y),DISCR_dkdy)
    i=i.subs(k,kC)
    
    i=i.subs(om.diff(y).diff(y),DISCR_d2omdy2)
    i=i.subs(om.diff(y),DISCR_domdy)
    i=i.subs(om,omC)
    
    i=i.subs(nu_T.diff(y).diff(y),DISCR_d2nutdy2)
    i=i.subs(nu_T.diff(y)        ,DISCR_dnutdy  )
    i=i.subs(nu_T,nutC)
    
    i=i.subs(beta.diff(y).diff(y),DISCR_d2BetaDy2)
    i=i.subs(beta.diff(y),DISCR_BetaDy)
    i=i.subs(beta,bC)
    return i
# Discretize equations
R_Udiscr =f_discretize(R_U )
R_kdiscr =f_discretize(R_k )
R_Omdiscr=f_discretize(R_Om)

# # # # Build the following required residual matrices:
# R,ui
# R,bi
# # # # Initialize residual vector and general variables
dimRn  =3
dimWvar=9
dim_Bvar=1
Rn_discr=MutableDenseNDimArray([R_Udiscr,R_kdiscr,R_Omdiscr],dimRn)
wVar=[uS,uC,uN,kS,kC,kN,omS,omC,omN] # Vector of DOF variables (u,k,om)
bVar=[bC] # Vector of "beta" coefficients (can be larger!!)

# Rn_discr_wi
Rn_discr_wi=MutableDenseNDimArray([0 for i in range(dimRn*dimWvar)],(dimRn,dimWvar))
for i in range(dimRn):
    for j in range(dimWvar):
        Rn_discr_wi[i,j]= Rn_discr[i].diff(wVar[j])
#
# Rn_discr_bi
Rn_discr_bi=MutableDenseNDimArray([0 for i in range(dimRn*dim_Bvar)],(dimRn,dim_Bvar))
for i in range(dimRn):
    for j in range(dim_Bvar):
        Rn_discr_bi[i,j]= Rn_discr[i].diff(bVar[j])
#
def f_removeGradsLaplas(i):
    i=i.replace(str(DISCR_d2udy2),'vDISCR_d2udy2')
    i=i.replace(str(DISCR_dudy  ),'vDISCR_dudy')
    
    i=i.replace(str(DISCR_d2kdy2),'vDISCR_d2kdy2')
    i=i.replace(str(DISCR_dkdy  ),'vDISCR_dkdy')
    
    i=i.replace(str(DISCR_d2omdy2),'vDISCR_d2omdy2')
    i=i.replace(str(DISCR_domdy  ),'vDISCR_domdy')
    
    i=i.replace(str(DISCR_d2nutdy2),'vDISCR_d2nutdy2')
    i=i.replace(str(DISCR_dnutdy  ),'vDISCR_dnutdy')
    i=i.replace(str(nutC  ),'vnutC')
    i=i.replace('**1.0','')
    return i
#
def fstrM(xx):
    vShape=xx.shape
    if len(vShape)==1:
        xxStr='np.array(['
        for j in range(vShape[0]):
            xxStr+='(' + str(xx[j]) + '),'
        xxStr+=']);'
    elif len(vShape)==2:
        xxStr='np.array(['
        for i in range(vShape[0]):
            xxStr+='['
            for j in range(vShape[1]):
                xxStr+='(' + str(xx[i,j]) + '),'
            xxStr+='],'
        xxStr+=']);'
    else:
        raise ValueError('fstrM: Matrix out of shape!!',vShape, str(xx)) 
    return xxStr.rstrip(';')
print('\n# PYTHON CODE.... (must be copy-pasted manually)')
vdRn_wi = str(f_removeGradsLaplas(fstrM(Rn_discr_wi)))
vdRn_bi = str(f_removeGradsLaplas(fstrM(Rn_discr_bi)))
print('\nvdRn_wi = %s' % vdRn_wi)
print('\nvdRn_bi = %s' % vdRn_bi)


# ------------------------------------------------------------------------
#                  Small Testing (new implementation)
# ------------------------------------------------------------------------

def testing():
    def quick_check(a,b):
        assert str(a).replace(' ','') == str(b).replace(' ','')
    
    quick_check(vdRn_wi,'np.array([[(cSd1* (vDISCR_dnutdy) + cSd2* (vnutC + nu_lam)), (cCd1* (vDISCR_dnutdy) + cCd2* (vnutC + nu_lam)), (cNd1* (vDISCR_dnutdy) + cNd2* (vnutC + nu_lam)), (C_mu* cSd1* (vDISCR_dudy)/ omega_S), (C_mu* cCd1* (vDISCR_dudy)/ omega_C + C_mu* (vDISCR_d2udy2)/ omega_C), (C_mu* cNd1* (vDISCR_dudy)/ omega_N), (-C_mu* cSd1* k_S* (vDISCR_dudy)/ omega_S**2), (-C_mu* cCd1* k_C* (vDISCR_dudy)/ omega_C**2 - C_mu* k_C* (vDISCR_d2udy2)/ omega_C**2), (-C_mu* cNd1* k_N* (vDISCR_dudy)/ omega_N**2), ], [(2* C_mu* beta_C* cSd1* k_C* (vDISCR_dudy)/ omega_C), (2* C_mu* beta_C* cCd1* k_C* (vDISCR_dudy)/ omega_C), (2* C_mu* beta_C* cNd1* k_C* (vDISCR_dudy)/ omega_C), (cSd1* sigma_k* (vDISCR_dkdy)/ omega_S + cSd2* (k_C* sigma_k/ omega_C + nu_lam) + cSd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (C_mu* beta_C* (vDISCR_dudy)**2/ omega_C - alpha_k* omega_C + cCd1* sigma_k* (vDISCR_dkdy)/ omega_C + cCd2* (k_C* sigma_k/ omega_C + nu_lam) + sigma_k* (vDISCR_d2kdy2)/ omega_C + cCd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (cNd1* sigma_k* (vDISCR_dkdy)/ omega_N + cNd2* (k_C* sigma_k/ omega_C + nu_lam) + cNd1* sigma_k* (vDISCR_dnutdy)/ C_mu), (-cSd1* k_S* sigma_k* (vDISCR_dkdy)/ omega_S**2), (-C_mu* beta_C* k_C* (vDISCR_dudy)**2/ omega_C**2 - alpha_k* k_C - cCd1* k_C* sigma_k* (vDISCR_dkdy)/ omega_C**2 - k_C* sigma_k* (vDISCR_d2kdy2)/ omega_C**2), (-cNd1* k_N* sigma_k* (vDISCR_dkdy)/ omega_N**2), ], [(2* cSd1* gamma* (vDISCR_dudy)), (2* cCd1* gamma* (vDISCR_dudy)), (2* cNd1* gamma* (vDISCR_dudy)), (cSd1* sigma_om* (vDISCR_domdy)/ omega_S + cSd1* sigma_d* (vDISCR_domdy)/ omega_C), (cCd1* sigma_d* (vDISCR_domdy)/ omega_C + cCd1* sigma_om* (vDISCR_domdy)/ omega_C + sigma_om* (vDISCR_d2omdy2)/ omega_C), (cNd1* sigma_om* (vDISCR_domdy)/ omega_N + cNd1* sigma_d* (vDISCR_domdy)/ omega_C), (-cSd1* k_S* sigma_om* (vDISCR_domdy)/ omega_S**2 + cSd1* sigma_d* (vDISCR_dkdy)/ omega_C + cSd2* (k_C* sigma_om/ omega_C + nu_lam) + cSd1* sigma_om* (vDISCR_dnutdy)/ C_mu), (-2* alpha_om* omega_C - cCd1* k_C* sigma_om* (vDISCR_domdy)/ omega_C**2 + cCd1* sigma_d* (vDISCR_dkdy)/ omega_C + cCd2* (k_C* sigma_om/ omega_C + nu_lam) - k_C* sigma_om* (vDISCR_d2omdy2)/ omega_C**2 - sigma_d* (vDISCR_dkdy)* (vDISCR_domdy)/ omega_C**2 + cCd1* sigma_om* (vDISCR_dnutdy)/ C_mu), (-cNd1* k_N* sigma_om* (vDISCR_domdy)/ omega_N**2 + cNd1* sigma_d* (vDISCR_dkdy)/ omega_C + cNd2* (k_C* sigma_om/ omega_C + nu_lam) + cNd1* sigma_om* (vDISCR_dnutdy)/ C_mu), ], ])')
    quick_check(vdRn_bi,'np.array([[(0),], [(C_mu*k_C*(vDISCR_dudy)**2/omega_C),], [(0),],])')

testing()