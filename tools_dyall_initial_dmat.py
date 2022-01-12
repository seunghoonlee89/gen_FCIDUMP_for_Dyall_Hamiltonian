import h5py
import numpy


def dumpERIs(ecore,int1e,int2e,fcidump):
   header=""" &FCI NORB=  12,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
   with open(fcidump,'w') as f:
     f.writelines(header)
     n = int1e.shape[0]
     # int2e
     nblk = 0
     np = n*(n+1)/2
     nq = np*(np+1)/2
     for i in range(n):
        for j in range(i+1):
           for k in range(i+1):
              if k == i: 	
                 lmax = j+1
              else:
                 lmax = k+1
              for l in range(lmax):
                 nblk += 1
                 line = str(int2e[i,j,k,l])\
                 + ' ' + str(i+1) \
                 + ' ' + str(j+1) \
                 + ' ' + str(k+1) \
                 + ' ' + str(l+1)+'\n'
                 if abs(int2e[i,j,k,l]) > 1e-10:
                     f.writelines(line)
     assert nq == nblk 
     # int1e
     nblk = 0 
     for i in range(n):
        for j in range(i+1):
            nblk += 1
            line = str(int1e[i,j])\
            + ' ' + str(i+1) \
            + ' ' + str(j+1) \
            + ' 0 0\n'
            if abs(int1e[i,j]) > 1e-10:
                f.writelines(line)
     assert np == nblk
     # ecore 
     line = str(ecore) + ' 0 0 0 0\n'
     f.writelines(line)
   print ('finish ',fcidump)
   return 0


def dump(info,ordering=None,fname='mole.h5',fcidump=False):
   ecore,int1e,int2e = info
   if ordering != None:
      int1e = int1e[numpy.ix_(ordering,ordering)].copy()
      int2e = int2e[numpy.ix_(ordering,ordering,ordering,ordering)].copy()
      dumpERIs(ecore,int1e,int2e)
      if fcidump: return 0  
   # dump information
   nbas = int1e.shape[0]
   sbas = nbas*2
   print ('\n[tools_itrf.dump] interface from FCIDUMP with nbas=',nbas)
   f = h5py.File(fname, "w")
   cal = f.create_dataset("cal",(1,),dtype='i')
   cal.attrs["nelec"] = 0.
   cal.attrs["sbas"]  = sbas
   cal.attrs["enuc"]  = 0.
   cal.attrs["ecor"]  = ecore
   cal.attrs["escf"]  = 0. # Not useful at all
   # Intergrals
   flter = 'lzf'
   # INT1e:
   h1e = numpy.zeros((sbas,sbas))
   h1e[0::2,0::2] = int1e # AA
   h1e[1::2,1::2] = int1e # BB
   # INT2e:
   h2e = numpy.zeros((sbas,sbas,sbas,sbas))
   h2e[0::2,0::2,0::2,0::2] = int2e # AAAA
   h2e[1::2,1::2,1::2,1::2] = int2e # BBBB
   h2e[0::2,0::2,1::2,1::2] = int2e # AABB
   h2e[1::2,1::2,0::2,0::2] = int2e # BBAA
   # <ij|kl> = [ik|jl]
   h2e = h2e.transpose(0,2,1,3)
   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
   h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
   int1e = f.create_dataset("int1e", data=h1e, compression=flter)
   int2e = f.create_dataset("int2e", data=h2e, compression=flter)
   # Occupation
   occun = numpy.zeros(sbas)
   orbsym = numpy.array([0]*sbas)
   spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
   f.create_dataset("occun",data=occun)
   f.create_dataset("orbsym",data=orbsym)
   f.create_dataset("spinsym",data=spinsym)
   f.close()
   print (' Successfully dump information for MPO-DMRG calculations! fname=',fname)
   print (' with ordering',ordering)
   return 0

def dump_fock(info,fcidump=False):
   ecore,int1e,int2e = info

   # occupation pattern for a major det
   occa = numpy.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1.])
   occb = numpy.array([1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

   # occupation of BS state
   h1e_fa = h1e.copy() 
   h1e_fb = h1e.copy() 
   h1e_fa += numpy.einsum('ijkk,k->ij', h2e, occa)
   h1e_fb += numpy.einsum('ijkk,k->ij', h2e, occb)
   
   vkaa = numpy.einsum('i,ikki,k->', occa, h2e, occa)
   vkbb = numpy.einsum('i,ikki,k->', occb, h2e, occb)
   e2 = (vjaa+vjbb+vjab+vjba-vkaa-vkbb) * .5

   if ordering != None:
      int1e = int1e[numpy.ix_(ordering,ordering)].copy()
      int2e = int2e[numpy.ix_(ordering,ordering,ordering,ordering)].copy()
      #dumpERIs(ecore,int1e,int2e)
      if fcidump: return 0  
   return 0

def dump_fock(h1e, h2e, occ_nat):
   fock_diag  = numpy.diag(h1e).copy()
   fock_diag += numpy.einsum('j,iijj->i', occ_nat, h2e)
   fock_diag -= numpy.einsum('j,ijji->i', occ_nat, h2e)*.5
   return numpy.eye(h1e.shape[0]) * fock_diag 

def dump_initial(info_cas, norb, nocc, thresh):
   ecore, fock, int2e = info_cas

   #coeff = i * 0.1
   idx = numpy.argsort(numpy.diag(fock))[::-1]
   print(numpy.diag(fock))
   print(numpy.diag(fock)[idx])
   for i in range(norb-nocc):
       fock[idx[i],idx[i]] += thresh

#   int2e = numpy.zeros((int2e_cas.shape)) 
   fcidump = 'FCIDUMPfockcas'
   dumpERIs(ecore,fock,int2e,fcidump)

def dump_cas_initial(info_cas, norb, nocc, ncas):
   rdm1 = numpy.load("rdm1_smallcas.npy")

   ecore_c, int1e_c, int2e_c = info_cas
   #ecore_c, int1e_c, int2e_c, _, _, _ = info_cas

   int1e = numpy.zeros(int1e_c.shape)
   int2e = numpy.zeros(int2e_c.shape)

   nvir = norb - nocc

   if ncas % 2 == 0:
       nc_cas = nocc - ncas//2
       noc_cas = nocc + ncas//2
   else:
       nc_cas = nocc - ncas//2 - 1
       noc_cas = nocc + ncas//2
   print('nc_cas, noc_cas',nc_cas, noc_cas)
   print(noc_cas-nc_cas, ncas)
   assert noc_cas-nc_cas == ncas

   fock  = numpy.zeros(int1e_c.shape)
   fock  = int1e_c.copy()
   # J and K btw C-C
   fock += 2. * numpy.einsum('ijkk->ij',    int2e_c[:,:,2:nc_cas,2:nc_cas])
   fock -=      numpy.einsum('ikkj->ij',    int2e_c[:,2:nc_cas,2:nc_cas,:])
   # J and K btw C-A
   fock +=      numpy.einsum('ijkl,kl->ij', int2e_c[:,:,:2,:2], rdm1[:2,:2])
   fock -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[:,:2,:2,:], rdm1[:2,:2])
   fock +=      numpy.einsum('ijkl,kl->ij', int2e_c[:,:,:2,nc_cas:noc_cas], rdm1[:2,2:])
   fock -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[:,:2,nc_cas:noc_cas,:], rdm1[:2,2:])
   fock +=      numpy.einsum('ijkl,kl->ij', int2e_c[:,:,nc_cas:noc_cas,:2], rdm1[2:,:2])
   fock -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[:,nc_cas:noc_cas,:2,:], rdm1[2:,:2])
   fock +=      numpy.einsum('ijkl,kl->ij', int2e_c[:,:,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[2:,2:])
   fock -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[:,nc_cas:noc_cas,nc_cas:noc_cas,:], rdm1[2:,2:])

   # Dyall hamiltonian
   # CORE
   int1e[2:nc_cas,2:nc_cas]  = fock[2:nc_cas,2:nc_cas].copy()
#   int1e = numpy.zeros(int1e_c.shape)
#   #numpy.fill_diagonal(int1e[2:nc_cas,2:nc_cas], numpy.diag(int1e_f[2:nc_cas,2:nc_cas]))
#   int1e[2:nc_cas,2:nc_cas]  = int1e_c[2:nc_cas,2:nc_cas].copy()
#   # J and K btw C-C
#   int1e[2:nc_cas,2:nc_cas] += 2. * numpy.einsum('ijkk->ij',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
#   int1e[2:nc_cas,2:nc_cas] -=      numpy.einsum('ikkj->ij',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
#   # J and K btw C-A
#   int1e[2:nc_cas,2:nc_cas] +=      numpy.einsum('ijkl,kl->ij', int2e_c[2:nc_cas,2:nc_cas,:2,:2], rdm1[:2,:2])
#   int1e[2:nc_cas,2:nc_cas] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[2:nc_cas,:2,:2,2:nc_cas], rdm1[:2,:2])
#
#   int1e[2:nc_cas,2:nc_cas] +=      numpy.einsum('ijkl,kl->ij', int2e_c[2:nc_cas,2:nc_cas,:2,nc_cas:noc_cas], rdm1[:2,2:])
#   int1e[2:nc_cas,2:nc_cas] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[2:nc_cas,:2,nc_cas:noc_cas,2:nc_cas], rdm1[:2,2:])
#
#   int1e[2:nc_cas,2:nc_cas] +=      numpy.einsum('ijkl,kl->ij', int2e_c[2:nc_cas,2:nc_cas,nc_cas:noc_cas,:2], rdm1[2:,:2])
#   int1e[2:nc_cas,2:nc_cas] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[2:nc_cas,nc_cas:noc_cas,:2,2:nc_cas], rdm1[2:,:2])
#
#   int1e[2:nc_cas,2:nc_cas] +=      numpy.einsum('ijkl,kl->ij', int2e_c[2:nc_cas,2:nc_cas,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[2:,2:])
#   int1e[2:nc_cas,2:nc_cas] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[2:nc_cas,nc_cas:noc_cas,nc_cas:noc_cas,2:nc_cas], rdm1[2:,2:])

   # ACTIVE
   int1e[:2,:2] = int1e_c[:2,:2].copy()
   int1e[:2,nc_cas:noc_cas] = int1e_c[:2,nc_cas:noc_cas].copy()
   int1e[nc_cas:noc_cas,:2] = int1e_c[nc_cas:noc_cas,:2].copy()
   int1e[nc_cas:noc_cas,nc_cas:noc_cas] = int1e_c[nc_cas:noc_cas,nc_cas:noc_cas].copy()
   # J and K btw A-C
   int1e[:2,:2] += 2. * numpy.einsum('ijkk->ij', int2e_c[:2,:2,2:nc_cas,2:nc_cas])
   int1e[:2,:2] -= 1. * numpy.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,:2])
   int1e[:2,nc_cas:noc_cas] += 2. * numpy.einsum('ijkk->ij', int2e_c[:2,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
   int1e[:2,nc_cas:noc_cas] -= 1. * numpy.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,nc_cas:noc_cas])
   int1e[nc_cas:noc_cas,:2] += 2. * numpy.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,:2,2:nc_cas,2:nc_cas])
   int1e[nc_cas:noc_cas,:2] -= 1. * numpy.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,:2])
   int1e[nc_cas:noc_cas,nc_cas:noc_cas] += 2. * numpy.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
   int1e[nc_cas:noc_cas,nc_cas:noc_cas] -= 1. * numpy.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,nc_cas:noc_cas])

   int2e[:2,:2,:2,:2] = int2e_c[:2,:2,:2,:2].copy()

   int2e[:2,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]
   int2e[nc_cas:noc_cas,:2,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,nc_cas:noc_cas]
   int2e[nc_cas:noc_cas,nc_cas:noc_cas,:2,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,nc_cas:noc_cas]
   int2e[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,:2]

   int2e[nc_cas:noc_cas,nc_cas:noc_cas,:2,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,:2]
   int2e[nc_cas:noc_cas,:2,nc_cas:noc_cas,:2] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,:2]
   int2e[nc_cas:noc_cas,:2,:2,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,:2,:2,nc_cas:noc_cas]
   int2e[:2,nc_cas:noc_cas,nc_cas:noc_cas,:2] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,:2]
   int2e[:2,nc_cas:noc_cas,:2,nc_cas:noc_cas] = int2e_c[:2,nc_cas:noc_cas,:2,nc_cas:noc_cas]
   int2e[:2,:2,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[:2,:2,nc_cas:noc_cas,nc_cas:noc_cas]

   int2e[nc_cas:noc_cas,:2,:2,:2] = int2e_c[nc_cas:noc_cas,:2,:2,:2]
   int2e[:2,nc_cas:noc_cas,:2,:2] = int2e_c[:2,nc_cas:noc_cas,:2,:2]
   int2e[:2,:2,nc_cas:noc_cas,:2] = int2e_c[:2,:2,nc_cas:noc_cas,:2]
   int2e[:2,:2,:2,nc_cas:noc_cas] = int2e_c[:2,:2,:2,nc_cas:noc_cas]

   int2e[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]

   # VIRTUAL
   #numpy.fill_diagonal(int1e[noc_cas:,noc_cas:], numpy.diag(int1e_f[noc_cas:,noc_cas:]))
   int1e[noc_cas:,noc_cas:]  = fock[noc_cas:,noc_cas:].copy()
#   int1e[noc_cas:,noc_cas:]  = int1e_c[noc_cas:,noc_cas:].copy()
#   # J and K btw V-C
#   int1e[noc_cas:,noc_cas:] += 2. * numpy.einsum('ijkk->ij',    int2e_c[noc_cas:,noc_cas:,2:nc_cas,2:nc_cas])
#   int1e[noc_cas:,noc_cas:] -=      numpy.einsum('ikkj->ij',    int2e_c[noc_cas:,2:nc_cas,2:nc_cas,noc_cas:])
#   # J and K btw V-A
##   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,:2,:2], rdm1[:2,:2])
##   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,:2,:2,noc_cas:], rdm1[:2,:2])
##   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[nc_cas:noc_cas,nc_cas:noc_cas])
##   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,nc_cas:noc_cas,nc_cas:noc_cas,noc_cas:], rdm1[nc_cas:noc_cas,nc_cas:noc_cas])
#   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,:2,:2], rdm1[:2,:2])
#   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,:2,:2,noc_cas:], rdm1[:2,:2])
#   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,:2,nc_cas:noc_cas], rdm1[:2,2:])
#   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,:2,nc_cas:noc_cas,noc_cas:], rdm1[:2,2:])
#   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,nc_cas:noc_cas,:2], rdm1[2:,:2])
#   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,nc_cas:noc_cas,:2,noc_cas:], rdm1[2:,:2])
#   int1e[noc_cas:,noc_cas:] +=      numpy.einsum('ijkl,kl->ij', int2e_c[noc_cas:,noc_cas:,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[2:,2:])
#   int1e[noc_cas:,noc_cas:] -= .5 * numpy.einsum('iklj,kl->ij', int2e_c[noc_cas:,nc_cas:noc_cas,nc_cas:noc_cas,noc_cas:], rdm1[2:,2:])

   # core energy
   ecore  = ecore_c
   ecore += 2. * numpy.einsum('ii->',      int1e_c[2:nc_cas,2:nc_cas])
   ecore += 2. * numpy.einsum('iikk->',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
   ecore -=      numpy.einsum('ikki->',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
   ecore -= 2. * numpy.einsum('ii->',      fock[2:nc_cas,2:nc_cas])

#   ecore -= 2. * numpy.einsum('iikl,kl->', int2e_c[2:nc_cas,2:nc_cas,:2,:2], rdm1[:2,:2])
#   ecore +=      numpy.einsum('ikli,kl->', int2e_c[2:nc_cas,:2,:2,2:nc_cas], rdm1[:2,:2])
#
#   ecore -= 2. * numpy.einsum('iikl,kl->', int2e_c[2:nc_cas,2:nc_cas,:2,nc_cas:noc_cas], rdm1[:2,2:])
#   ecore +=      numpy.einsum('ikli,kl->', int2e_c[2:nc_cas,:2,nc_cas:noc_cas,2:nc_cas], rdm1[:2,2:])
#
#   ecore -= 2. * numpy.einsum('iikl,kl->', int2e_c[2:nc_cas,2:nc_cas,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[2:,2:])
#   ecore +=      numpy.einsum('ikli,kl->', int2e_c[2:nc_cas,nc_cas:noc_cas,nc_cas:noc_cas,2:nc_cas], rdm1[2:,2:])
#
#   ecore -= 2. * numpy.einsum('iikl,kl->', int2e_c[2:nc_cas,2:nc_cas,nc_cas:noc_cas,:2], rdm1[2:,:2])
#   ecore +=      numpy.einsum('ikli,kl->', int2e_c[2:nc_cas,nc_cas:noc_cas,:2,2:nc_cas], rdm1[2:,:2])

   #coeff = i * 0.1
   #idx = numpy.argsort(numpy.diag(fock))[::-1]
   #print(numpy.diag(fock))
   #print(numpy.diag(fock)[idx])

#   int2e = numpy.zeros((int2e_cas.shape)) 
   #fcidump = 'FCIDUMPfockcas_ncas%d'%(ncas)
   fcidump = 'FCIDUMPfockcas'
   dumpERIs(ecore,int1e,int2e,fcidump)

if __name__ == '__main__':
   import sys
   n_cas = int(sys.argv[1]) - 2
   #n_det = int(sys.argv[2])
   print('n_cas=', n_cas+2)
   import tools_io
   info_cas  = tools_io.loadERIs('FCIDUMPcas')
   #info_fock = tools_io.loadERIs('FCIDUMPfockcas_occ')
   nelec = 14 
   norb  = 12 
   nocc  = nelec // 2 

#   # find det_occ
#   import numpy
#   ci = numpy.load("fcivec1000.npy")[0]
#   from pyscf.fci import cistring
#   
#   import tools_io
#   neleca = 7 
#   nelecb = 7 
#   norb   = 12 
#   tol    = 0.0 
#   
#   na = cistring.num_strings(norb, neleca)
#   nb = cistring.num_strings(norb, nelecb)
#   print(ci.shape, na, nb)
#   assert(ci.shape == (na, nb))
#   
#   addra, addrb = numpy.where(abs(ci) > tol)
#   strsa = cistring.addrs2str(norb, neleca, addra)
#   strsb = cistring.addrs2str(norb, nelecb, addrb)
#   occa = cistring._strs2occslst(strsa, norb)
#   occb = cistring._strs2occslst(strsb, norb)
#   
#   ci_tol = ci[addra,addrb]
#   overlap= numpy.square(ci_tol)
#   idx  = numpy.argsort(overlap)
#   overlap  = overlap[idx][::-1]
#   ci_tol = ci_tol[idx][::-1]
#   occa = occa[idx][::-1]
#   occb = occb[idx][::-1]
#   
#   samp = [n_det]
#   
#   sequence = [i for i in range(norb)]
#   epsilon = 0.5
#   iit = 0
#   for i in range(occa.shape[0]):
#       if numpy.array_equal(occa[i], occb[i]):
#           iit += 1
#           if iit not in samp: continue
#   
#           occ_det = occa[i]

#   #print(occ_det)
#   occ_det = [0,1,2,3,4,5,8]
#   sequence = [i for i in range(norb)]
#   thresh_list = list(set(sequence)-set(occ_det))

   #occ_nat = [1.965, 1.958 ,1.414 ,1.263 ,1.446 ,1.094 ,0.843 ,0.909 ,1.124 ,0.896 ,0.561 ,0.526]
   #ordering = [4, 5, 3, 6, 1, 2, 7, 8 ]
   #ordering = [i-1 for i in ordering]
   #print (len(ordering))
   #print ('ordering=',ordering)

   dump_cas_initial(info_cas, norb, nocc, n_cas) 

