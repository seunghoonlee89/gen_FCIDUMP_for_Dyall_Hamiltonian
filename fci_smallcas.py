#from __future__ import print_function
import numpy
from pyscf import gto,scf
from pyscf.tools.fcidump import write_head
import tools_io
import sys

def fes(bond, nroots):
    #==================================================================
    # MOLECULE
    #==================================================================
    mol = gto.Mole()
    mol.verbose = 5 #6
    
    #==================================================================
    # Coordinates and basis
    #==================================================================
    molname = 'fes' #be' #h2cluster'#h2o3' #c2'
    
    if molname == 'fes':
       mol.atom ='''
 Fe                 5.22000000    1.05000000   -7.95000000
 S                  3.86000000   -0.28000000   -9.06000000
 S                  5.00000000    0.95000000   -5.66000000
 S                  4.77000000    3.18000000   -8.74000000
 S                  7.23000000    0.28000000   -8.38000000
 Fe                 5.88000000   -1.05000000   -9.49000000
 S                  6.10000000   -0.95000000  -11.79000000
 S                  6.33000000   -3.18000000   -8.71000000
 C                  6.00000000    4.34000000   -8.17000000
 H                  6.46000000    4.81000000   -9.01000000
 H                  5.53000000    5.08000000   -7.55000000
 H                  6.74000000    3.82000000   -7.60000000
 C                  3.33000000    1.31000000   -5.18000000
 H                  2.71000000    0.46000000   -5.37000000
 H                  3.30000000    1.54000000   -4.13000000
 H                  2.97000000    2.15000000   -5.73000000
 C                  5.10000000   -4.34000000   -9.28000000
 H                  5.56000000   -5.05000000   -9.93000000
 H                  4.67000000   -4.84000000   -8.44000000
 H                  4.34000000   -3.81000000   -9.81000000
 C                  7.77000000   -1.31000000  -12.27000000
 H                  7.84000000   -1.35000000  -13.34000000
 H                  8.42000000   -0.54000000  -11.90000000
 H                  8.06000000   -2.25000000  -11.86000000
'''
       mol.basis = 'tzp-dkh'
       mol.charge = -2
       mol.spin = 0 
       mol.max_memory = 50000 # 125 g
       mol.build()
    
    mol.symmetry = False #True
    mol.build()
    
    #==================================================================
    # SCF
    #==================================================================
    #mf = scf.sfx2c(scf.RKS(mol))
    mf = scf.RHF(mol)
   
    from pyscf import mcscf, fci
    from functools import reduce
    from pyscf.tools import fcidump
    from pyscf import ao2mo
    
    #ecore, h1e, eri, norb, nelec, ms = tools_io.loadERIs('../FCIDUMP/FCIDUMPsmallcas')
    ecore, h1e, eri, norb, nelec, ms = tools_io.loadERIs2('FCIDUMPsmallcas')
    neleca = (nelec+ms)//2
    nelecb = (nelec-ms)//2

    #norb = h1e.shape[0]
    #nelec= 20 
#    import math
#    def comb(n: int, k: int) -> int:
#        n_fac = math.factorial(n)
#        k_fac = math.factorial(k)
#        n_minus_k_fac = math.factorial(n - k)
#        return n_fac/(k_fac*n_minus_k_fac)
#
# direct FCI    
#    from pyscf import fci
#    fci_space = int(comb(norb,nelec//2) ** 2)
#    print('tot nroot',fci_space)
#    H_fci = fci.direct_spin1.pspace(h1e, eri, norb, nelec, np=fci_space)[1]
#    e_all, v_all = numpy.linalg.eigh(H_fci)
#    idx = numpy.argsort(e_all)   
#    e_all = e_all[idx]
#
#    numpy.save("fci.npy", e_all)


    #nroots = 2
    spin = mol.spin/2
    #print('S^2 = ',spin*(spin+1))
    mci = fci.direct_spin1.FCI(mol)
    #mci = fci.direct_spin0.FCISolver(mol)
    mci.spin = 0 
    mci.conv_tol = 1e-8
    mci.max_cycle = 2000
    mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
    # ss = s*(s+1)
    # s = 0.5, ss = 0.5*1.5 = 0.75 
    e, fcivec = mci.kernel(h1e, eri, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000)
    e = numpy.array(e)
    numpy.save("E_smallcas.npy", e+ecore)
    numpy.save("fcivec_smallcas.npy", fcivec[0])
    rdm1, rdm2 = mci.trans_rdm12(fcivec[0], fcivec[0], norb, nelec) 
    numpy.save("rdm1_smallcas.npy", rdm1)

    for i in range(nroots):
        first = True
        res = mci.large_ci(fcivec[i], norb, nelec, 0.2, return_strs=False)
        ref_a = [i for i in range(neleca)]
        ref_b = [i for i in range(nelecb)]

        res_unzip = list(zip(*res))
        if len(res_unzip[0]) > 1:
            idx = numpy.argsort(numpy.abs(res_unzip[0]))[::-1]
            res_unzip[0] = numpy.array(res_unzip[0])[idx] 
            res_unzip[1] = numpy.array(res_unzip[1])[idx] 
            res_unzip[2] = numpy.array(res_unzip[2])[idx] 

        for c,ia,ib in zip(res_unzip[0], res_unzip[1], res_unzip[2]):
            ah = []
            bh = []
            ap = []
            bp = []
            for a in ia:
                if a not in ref_a:
                    ap.append(a)  
            for a in ref_a:
                if a not in ia:
                    ah.append(a)  
            for a in ib:
                if a not in ref_b:
                    bp.append(a)  
            for a in ref_b:
                if a not in ib:
                    bh.append(a)  
            if first:
                print('')
                print('th excited state')
                print(e[i]+ecore)
                print('2S+1') 
                print(fci.spin_op.spin_square0(fcivec[i], norb, nelec)[1])
                print('excitation                                        CI coefficient')
                first = False

            print('  %-20s %-30s %.12f'%( ia, ib, c))
            for h, p in zip(ah, ap):
                print('a %d -> %d'%(h, p),end=', ')
            for h, p in zip(bh, bp):
                print('b %d -> %d'%(h, p),end=', ')
            print('%-*s'%(50-4*(len(ap)+len(ah)+len(bp)+len(bh)),''),end='')
            print('%.12f'%(c))

     
nroots = 3 
results = []
import numpy
bonds = [2.0] 
for bond in bonds:
    fes(bond, nroots)



