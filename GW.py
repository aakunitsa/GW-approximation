# A simple implementation of GW@DFT class; meant to be used to run GW on small molecules
# using RI/CD GW algorithm

import psi4
import numpy as np
import scipy as sp


class GW_DFT:

    def __init__(self, wfn, mol, gw_par):

        # wfn - Psi4 w.f. object from SCF calculation
        # mol - Psi4 molecule object
        # gw_par - is a dictionary with GW calculation parameters
        #          such as the number of states or the number of
        #          omega sampling points

        self.scf_wfn = wfn
        self.mol = mol
        self.gw_par = gw_par

        assert hasattr(self, 'scf_wfn')
        assert hasattr(self, 'mol')
        assert hasattr(self, 'gw_par')


        self._init_sys_params() # sets some basic system parameters
        self._gen_ri_ints()     # generates integrals for RI-GW and
        # RI integrals are now available in self.nmR

        # Generate modified Gauss-Legendre grid for the frequency integration on imaginary axis

        self.gl_npoint =  100 if not 'gl_npoint' in gw_par.keys() else gw_par['gl_npoint']
        self._gen_gaussleg()

        # Setup exchange-correlation kernel

        self._XC()

        # set GW calculation parameters

        # parameters of the self-energy calculation
        nomega_sigma =  501 if not 'nomega_sigma' in gw_par.keys() else gw_par['nomega_sigma']
        step_sigma = 0.01 if not 'step_sigma' in gw_par.keys() else  gw_par['step_sigma']

        # Quasi-particle states
        self.no_qp = self.nocc if not 'no_qp' in gw_par.keys() else gw_par['no_qp'] # Number of hole states
        self.nv_qp = 0 if not 'nv_qp' in gw_par.keys() else gw_par['nv_qp']             # Number of particle states
        self.eta = 1e-3 if not 'eta' in gw_par.keys() else gw_par['eta'] # Default eta=1e-3 is recommended by Bruneval

        # Algorithms
        self.debug = False if not 'debug' in gw_par.keys() else gw_par['debug']
        self.low_mem = True if not 'low_mem' in gw_par.keys() else gw_par['low_mem']

        # Screened Coulomb interaction
        self.analytic_W = False if not 'analytic_W' in gw_par.keys() else gw_par['analytic_W']

        if self.debug:
            print("Running in debug mode!")
        else:
            print("Running in production mode!")


        # Quick sanity check

        assert self.no_qp <= self.nocc and self.nv_qp <= self.nvir

        # ### GW calculation starts here ###

        # create an array of sampling frequencies similar to MolGW

        nomega_grid = nomega_sigma // 2 # note this is a truncation (aka integer) division
        omega_grid = np.array(range(-nomega_grid, nomega_grid + 1)) * step_sigma


        # sampling energies for all the states so we could calculate the self-energy matrix (broadcasting)

        omega_grid_all = omega_grid + self.eps[self.nocc - self.no_qp:self.nocc + self.nv_qp].reshape((-1, 1))
        assert omega_grid_all.shape == (self.no_qp + self.nv_qp, 2*nomega_grid + 1)
        print("Shape of the omega_grid_all is ", omega_grid_all.shape)
        self.omega_grid_all = np.copy(omega_grid_all)

        method = 'contour deformation'
        print("Caculating GW self-energy via %s" % (method))

        if self.analytic_W:
            print("Analytic W has been requested; performing RPA calculation")
            self._RPA(gw_par)

        Sigma_c_grid = self._calculate_iGW(omega_grid_all) # self-energy matrix
        print("Finished calculating self-energy")
        Sigma_x = np.zeros(self.no_qp + self.nv_qp)
        I = np.einsum("nmQ, mnQ->nm", self.nmR[self.nocc - self.no_qp:self.nocc + self.nv_qp, :self.nocc, :], self.nmR[:self.nocc, self.nocc - self.no_qp:self.nocc + self.nv_qp, :])
        Sigma_x = -np.einsum("nm->n", I)

        self.Sigma_c_grid = np.copy(Sigma_c_grid)

        # Apply solvers; Similar to MolGW - linear & graphic solutions

        print("Performing one-shot G0W0")

        qp_molgw_lin_ = np.zeros(self.no_qp + self.nv_qp)

        # Calculate pole strengths by performing numerical derivative on the omega grid

        zz = np.real(Sigma_c_grid[:, nomega_grid + 1] - Sigma_c_grid[:, nomega_grid - 1]) / (omega_grid[nomega_grid + 1] - omega_grid[nomega_grid - 1])
        zz = 1. / (1. - zz)
        zz[zz <= 0.0] = 0.0
        zz[zz >= 1.0] = 1.0


        xc_contr = (1. - self.alpha) * Sigma_x[self.nocc - self.no_qp:self.nocc + self.nv_qp] - np.diag(self.Vxc)[self.nocc - self.no_qp:self.nocc + self.nv_qp]
        print("SigX - Vxc")
        print(xc_contr)
        self.Sigma_x_Vxc = np.copy(xc_contr)

        qp_molgw_lin_ = self.eps[self.nocc - self.no_qp:self.nocc + self.nv_qp] + zz * (np.real(Sigma_c_grid[:, nomega_grid]) + xc_contr)
        #print(qp_molgw_lin_.shape)

        print("Perfoming graphic solution of the inverse Dyson equation")

        # both rhs and lhs of the QP equation have been calculated above

        qp_molgw_graph_ = np.copy(self.eps[self.nocc - self.no_qp:self.nocc + self.nv_qp])
        zz_graph = np.zeros(self.no_qp + self.nv_qp)
        self.graph_solver_data = {} # Format: state = [[e1, e2, ...], [z1, z2, ...]]

        for state in range(self.no_qp + self.nv_qp):
            z , e = self._find_fixed_point(omega_grid_all[state], np.real(Sigma_c_grid[state, :]) + self.eps[state + self.nocc - self.no_qp] + (1. - self.alpha) * Sigma_x[state + self.nocc - self.no_qp] - np.diag(self.Vxc)[state + self.nocc - self.no_qp])
            if z[0] < 1e-6:
                print("Graphical solver failed for state %d" % (state + 1))
            # Do nothing since the array cell already contains HF orbital energy
            else:
                qp_molgw_graph_[state] = e[0]
                zz_graph[state] = z[0]
                # Save all the solutions to graph_solver_data in case wrong  solution 
                # has the largest Z
                self.graph_solver_data[state] = [e, z]

        self.zz = np.copy(zz)
        self.qp_molgw_lin_ = np.copy(qp_molgw_lin_)
        self.qp_molgw_graph_ = np.copy(qp_molgw_graph_)

        print("Done!")

    def print_summary(self):

        Ha2eV = psi4.constants.hartree2ev

        print("E^lin, eV  E^graph, eV  Z ")
        for i in range(self.no_qp + self.nv_qp):
            print("%13.6f  %13.6f  %13.6f" % (self.qp_molgw_lin_[i]*Ha2eV, self.qp_molgw_graph_[i]*Ha2eV, self.zz[i]))

        print("Graphical solver printout")
        for s in self.graph_solver_data:
            print("State %d" % (s))
            print("E_qp, eV   Z")
            e_vals, z_vals = self.graph_solver_data[s]
            for e, z in zip(e_vals, z_vals):
                print("%13.6f %13.6f" % (e * Ha2eV, z))


    def int_dump(self, filename='INTDUMP'):

        output = open(filename, 'w')
        print("Saving inegrals and SCF data to a disk file...")
        naux = self.nmR.shape[2]
        output("%5d %5d" % (self.nbf, naux))

        # Write orbitals
        for e in self.eps:
            output.write("%29.20f" % e)

        for n in range(self.nbf):
            for m in range(self.nbf):
                for R in range(naux):
                    output.write("%5d %5d %5d %29.20f" % (n, m, R, self.nmR))

        print("Orbital energies and RI integrals were saved to file %s" % (filename))

        output.close()


    def _init_sys_params(self):

        self.nocc = self.scf_wfn.nalpha()
        self.nbf = self.scf_wfn.nmo()
        self.nvir = self.nbf - self.nocc
        self.C = self.scf_wfn.Ca()

        self.Cocc = self.scf_wfn.Ca_subset("AO", "OCC")

        self.eps = np.asarray(self.scf_wfn.epsilon_a())

        # print a quick summary
        print("Number of basis functions: ", self.nbf)
        print("occ/virt: %d/%d" % (self.nocc, self.nvir))

    def _XC(self):

        assert hasattr(self, 'scf_wfn')


        # The function constructs exchange-corrlation
        # potential matrix and extracts some other
        # relevant data from PSI4 objects
        # It is assumed that we are working with the closed-shell
        # reference

        Va = np.asarray(self.scf_wfn.Va())
        self.Vxc = np.einsum("ia, ij, jb-> ab", self.C, Va, self.C)
        self.alpha = self.scf_wfn.V_potential().functional().x_alpha() # Fraction of HF exchange

        print("Fraction of HF exchange is %6.3f" % (self.alpha))

        # will also need the integrals of exchange-correlation kernel
        # for fully analytic calculation but that will be implemented later

    def _gen_ri_ints(self):

        # MO coefficients
        C = np.asarray(self.C)

        # Extract basis set from the wfn object
        orb = self.scf_wfn.basisset()

        # Determine RI parameters 
        ri_type = "RIFIT" if not 'ri_type' in self.gw_par.keys() else self.gw_par['ri_type']
        ri_basis = str(orb.name()) if not 'ri_basis' in self.gw_par.keys() else self.gw_par['ri_basis'] 

        # Sanity check 
        assert ri_type in ["RIFIT", "JKFIT"]

        print("Attempting to create RI basis set for %s (%s)... " % (ri_basis, ri_type))

        # Build auxiliary basis set
        #aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", orb.name())
        #aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "RIFIT", orb.name())
        aux = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", ri_type, ri_basis)

        # From Psi4 doc as of March, 2019 (http://www.psicode.org/psi4manual/1.2/psi4api.html#psi4.core.BasisSet.zero_ao_basis_set):
        # Returns a BasisSet object that actually has a single s-function at
        # the origin with an exponent of 0.0 and contraction of 1.0.
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        # Create a MintsHelper Instance
        mints = psi4.core.MintsHelper(orb)

        # Build (pq|P) raw 3-index ERIs, dimension (nbf, nbf, Naux, 1)
        pqP = mints.ao_eri(orb, orb, aux, zero_bas)

        # Build and invert the metric
        metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
        metric.power(-0.5, 1.e-14)

        # Remove the dimensions of size 1
        pqP = np.squeeze(pqP)
        metric = np.squeeze(metric)

        # Transform (pq|P) to obtain (nm|P) in molecular orbital basis
        nmP = np.einsum("pn, qm, pqR-> nmR", C, C, pqP)

        # Contract with the inverse square root of the metric tensor
        self.nmR = np.einsum( "nmP, PR-> nmR", nmP, metric)

        print("Auxiliary basis set has been generated!")
        print("Number of auxiliary basis functions: ", self.nmR.shape[2])

    def _gen_gaussleg(self):

        x, w = np.polynomial.legendre.leggauss(self.gl_npoint)
        self.gl_x = (1. + x) / (1. - x)
        self.gl_w = 2. * w / (1. - x)**2


    def _find_fixed_point(self, lhs, rhs):
        # This function returns an array of fixed points and correspoinding pole strengths
        # Its application can be vectorized using strandard NumPy np.vectorize

        assert lhs.shape == rhs.shape

        # Maximum number of fixed points (same as in MolGW)
        nfp_max = 4

        # Pole strength threshold
        pthresh = 1e-5

        # Arrays of f.p. energies and  Z
        zfp = np.zeros(nfp_max)
        zfp[:] = -1.0
        efp = np.zeros(nfp_max)

        # Auxiliary index array
        idx = np.arange(nfp_max)

        n = len(lhs)
        ifixed = 0
        g = rhs - lhs

        # loop over grid points excluding the last one

        for i in range(n - 1):
            if g[i] * g[i + 1] < 0.0:
                #print("Fixed point found betwenn %13.6f and %13.6f eV! " % (lhs[i] * Ha2eV, lhs[i+1] * Ha2eV))
                z_zero = 1. / ( 1. - ( g[i+1] - g[i] ) / ( lhs[i+1] - lhs[i] ) )
                if z_zero < pthresh:
                    continue
                # Do some bookkeeping; the code looks ugly but that is exactly what F.Bruneval has in MolGW package

                if z_zero > zfp[-1]:
                    jfixed = np.min(idx[z_zero > zfp])
                    zfp[jfixed + 1:] = zfp[jfixed:nfp_max - 1]
                    efp[jfixed + 1:] = efp[jfixed:nfp_max - 1]
                    zfp[jfixed] = z_zero
                    # Perfom linear interpolation to find the root
                    zeta = (g[i + 1] - g[i]) / (lhs[i + 1] - lhs[i])
                    efp[jfixed] = lhs[i] - g[i] / zeta

        #print("Graphical solver concluded operation")
        return (zfp, efp)

    def _RPA(self, gw_par):
                
        nocc = self.nocc
        nvir = self.nvir
        
        # Diagonal \epsilon_a - \epsilon_i
        eps_diag = self.eps[nocc:].reshape(-1, 1) - self.eps[:nocc]
        assert eps_diag.shape == (nvir, nocc)
        
        # A^{+} + B^{+}
        ApB = np.einsum("ij,ab,ai -> iajb", np.eye(nocc), np.eye(nvir), eps_diag) + 4. * np.einsum("iaQ, jbQ->iajb", self.nmR[:nocc, nocc:], self.nmR[:nocc, nocc:])
        
        ApB = ApB.reshape((nocc*nvir, nocc*nvir)) 
        # since nD numpy arrays have C-style memroy layout the occupied orbital inedex changes slower than the virtual one

        # Diagonal of  A^{+} - B^{+}
        AmB_diag = eps_diag.T.reshape((1, -1))
        AmB_diag = np.diag(AmB_diag[0,:])

        assert AmB_diag.shape == ApB.shape

        # Form C matrix (as one usually does when solving RPA eigenvalue problem)
        C_ = np.einsum("ij,jk,kl->il", np.sqrt(AmB_diag), ApB, np.sqrt(AmB_diag))
        
        # Solve for the excitation energies and calculate X + Y eigenvectors
        
        omega2, Z = np.linalg.eigh(C_)
        self.omega_s = np.sqrt(omega2)
        self.xpy = np.einsum("ij,jk,kl->il", np.sqrt(AmB_diag), Z, np.diag(1./np.sqrt(self.omega_s)))
        
        if self.debug:
            print("RPA excitation energies:")
            print(self.omega_s)

    def _calculate_iGW(self, omega_grid_all):

        nocc = self.nocc
        nvir = self.nvir
        eps = self.eps

        no_qp = self.no_qp
        nv_qp = self.nv_qp
        nbf = self.nbf

        e_fermi = (eps[nocc - 1] + eps[nocc]) / 2.
        ngrid = omega_grid_all.shape[1]

        naux = self.nmR.shape[2]

        assert omega_grid_all.shape == (no_qp + nv_qp, ngrid)

        # Self-energy calculation will be performed via contour deformation
        # analytic continuation can be implemented later for comparison and
        # benchmarking

        e_ai = eps[nocc:, np.newaxis] - eps[np.newaxis, :nocc]
        im_grid = self.gl_x * 1.j

        # Calculate some intermediates for the imaginary time integration
        f = np.ones(nbf)
        f[eps > e_fermi] = -1.
        assert np.sum(f) == nocc - nvir
        complex_eps = eps + 0.5j * self.eta * f

        Wnm_im_grid = np.zeros((no_qp + nv_qp, nbf, len(self.gl_x)))
        omega_rts = np.zeros(1) 
        # Calculate Wnm on the imaginary frequency grid
        if self.analytic_W:

            ### Aanalytic calculation of W on imaginary frequency grid

            # Omega tensors; Will be reused later to calculate the residue term
            i_rtia = np.einsum("iaQ, rtQ ->rtia", self.nmR[:nocc, nocc:, :], self.nmR)
            i_rtia = i_rtia.reshape((nbf, nbf, nocc*nvir))
            omega_rts = np.sqrt(2.) * np.einsum("rtk, ks->rts", i_rtia, self.xpy)
            print("Shape of omega tensor is ", omega_rts.shape)

            Ds_p = self.omega_s.reshape((-1, 1)) + im_grid  - 0.5j*self.eta
            Ds_m = -self.omega_s.reshape((-1, 1)) + im_grid  + 0.5j*self.eta

            assert Ds_p.shape == Ds_m.shape and Ds_m.shape == (len(self.omega_s), len(self.gl_x))

            Wnm_im_grid = np.einsum("nms, sg, nms -> nmg",omega_rts[nocc - no_qp:nocc + nv_qp,:,:], 1./Ds_m - 1./Ds_p, omega_rts[nocc - no_qp:nocc + nv_qp,:,:])
            #print(Wnm_im_grid.shape)
            #print( (no_qp + nv_qp, nbf, len(self.gl_x)) )
            assert Wnm_im_grid.shape == (no_qp + nv_qp, nbf, len(self.gl_x))

        else:

            O_ = self.nmR[nocc-no_qp:nocc+nv_qp, :, :]
            dp_ = im_grid[:,np.newaxis, np.newaxis] + e_ai[np.newaxis, :, :] - 0.5j * self.eta
            dm_ = im_grid[:,np.newaxis, np.newaxis] - e_ai[np.newaxis, :, :] + 0.5j * self.eta
            if self.debug:
                dp_debug = np.zeros((self.gl_npoint, nvir, nocc), dtype=np.complex128)
                dm_debug = np.zeros((self.gl_npoint, nvir, nocc), dtype=np.complex128)
                for idx, grid_point in enumerate(im_grid):
                    tmp_p = grid_point + e_ai - 0.5j * self.eta
                    tmp_m = grid_point - e_ai + 0.5j * self.eta
                    dp_debug[idx, :, :] = tmp_p
                    dm_debug[idx, :, :] = tmp_m

                assert np.allclose(dp_debug, dp_) and np.allclose(dm_debug, dm_)


            id_pq = np.eye(naux)
            assert id_pq.shape == (naux, naux) and np.all(np.diag(id_pq) == np.ones(naux))
            #Ppq_ = np.einsum("iaP, gai, iaQ->gPQ", self.nmR[:nocc, nocc:,:], 1./dm_ - 1./dp_, self.nmR[:nocc,nocc:,:])
            Ppq_ = 2. * np.einsum("iaP, gai, iaQ->gPQ", self.nmR[:nocc, nocc:,:], 1./dm_ - 1./dp_, self.nmR[:nocc,nocc:,:])
    
            if self.debug:
                Ppq_debug = np.zeros((self.gl_npoint, naux, naux), dtype=np.complex128)
                tmp_O = self.nmR[:nocc,nocc:,:]
                for idx, grid_point in enumerate(im_grid):
                    #Ppq_debug[idx, :,:] = np.einsum("iaP, ai, iaQ->PQ", tmp_O, 1./dm_[idx,:,:] - 1./dp_[idx,:,:], tmp_O)
                    Ppq_debug[idx, :,:] = 2. * np.einsum("iaP, ai, iaQ->PQ", tmp_O, 1./dm_[idx,:,:] - 1./dp_[idx,:,:], tmp_O)
    
                assert np.allclose(Ppq_debug, Ppq_)


            assert Ppq_.shape == (len(im_grid), naux, naux)
            Wnm_im_grid = np.einsum("nmP, lPQ, nmQ -> nml", O_, (np.linalg.inv(id_pq[np.newaxis, :,:] - Ppq_) - id_pq[np.newaxis, :,:]), O_)
            assert Wnm_im_grid.shape == (no_qp + nv_qp, nbf, len(self.gl_x))

            # Check if matrix inverse is numerically accurate

            if self.debug:
                inv_thresh = 1e-12
                for idx, grid_point in enumerate(im_grid):
                    tmp1 = id_pq - Ppq_[idx, :,:]
                    tmp2 = np.linalg.inv(tmp1)
                    tmp3 = np.dot(tmp1, tmp2)
                    max_err = np.max(np.abs(np.real(tmp3) - id_pq))
                    max_err_im = np.max(np.abs(np.imag(tmp3)))
                    if max_err > inv_thresh or max_err_im > inv_thresh:
                        print("Matrix inverse failed when calculating the integral term!")
                        print(max_err)
                        print(max_err_im)


        # ### GW self-energy calculation via contour deformation
        # Inform the user about the amout of memory required for the calculation
        # with the current implementation (residue term is the most expensive one)
        # This excules the amout of memory needed to store the target objects

        mem_int = (nocc * nvir * (no_qp + nv_qp) + 4 * (nocc * nvir * len(self.gl_x)) + 2 * (len(im_grid) * naux * naux))
        mem_res = 4. * ngrid * nbf * nocc * nvir + 2 * ngrid * nbf * naux * naux + nbf * naux + naux**2
        if self.low_mem:
            mem_res = 4. * ngrid * nocc * nvir + naux + naux**2 + 2 * ngrid * naux**2

        print("Calculation of the integral term requires %8.3f Gb" %(mem_int * 8e-9)) # Each standard double is 8 bytes
        print("Calculation of the residue term requires  %8.3f Gb" %(mem_res * 8e-9))
        if self.low_mem:
            print("Using low-memory algorithm")

        In = np.zeros(omega_grid_all.shape, dtype=np.complex128) # Integral term
        Rn = np.zeros(omega_grid_all.shape, dtype=np.complex128) # Residue term

        # Integral term
        for qp in range(no_qp + nv_qp):
            # Calculate GF denominators = omega + 1.j omega_prime - eps_m \pm i eta
            qp_grid = np.copy(omega_grid_all[qp, :])

            Dgf_p = qp_grid[:, np.newaxis, np.newaxis]  - complex_eps[np.newaxis, :, np.newaxis] + im_grid[np.newaxis,np.newaxis, :]
            Dgf_m = qp_grid[:, np.newaxis, np.newaxis]  - complex_eps[np.newaxis, :, np.newaxis] - im_grid[np.newaxis,np.newaxis, :]

            if self.debug:
                # Print some diagnostic information about the denominators
                thresh = 1e-10
                Dgf_p_min = np.min(np.absolute(Dgf_p))
                Dgf_m_min = np.min(np.absolute(Dgf_m))

                if Dgf_p_min < thresh or Dgf_m_min < thresh:
                    print("Small denominator detected when calculating the integral term! Diagonstic info is printed below:")
                    print(Dgf_p_min_re)
                    print(Dgf_m_min_re)



            assert Dgf_p.shape == (ngrid, nbf, len(self.gl_x)) and Dgf_m.shape == (ngrid, nbf, len(self.gl_x))
            Wnm_tmp = np.copy(Wnm_im_grid[qp,:,:])
            I_term = 1. / (2.*np.pi) * np.einsum("fmg, g->f", Wnm_tmp[np.newaxis, :,:]* (1./ Dgf_p + 1./Dgf_m), self.gl_w)
            In[qp, :] = np.copy(I_term)
            del Dgf_p
            del Dgf_m
            del Wnm_tmp
            del I_term

            # Residue term
            # Caculate Wnm and f vector for a give quasi-particle
            # Not memory efficient; just for testing

            #offset_complex_eps = complex_eps[:, np.newaxis] - qp_grid
            offset_complex_eps = np.abs(eps[:, np.newaxis] - qp_grid) + 0.5j * self.eta
            assert offset_complex_eps.shape == (nbf, ngrid)

            Wnm_4res = np.zeros((nbf, ngrid), dtype=np.complex128)

            fill_factors = np.zeros((ngrid, nbf))
            #mask_vir = np.logical_and(qp_grid[:,np.newaxis] > eps[np.newaxis, :], eps.reshape((1, -1)) > e_fermi)
            #mask_occ = np.logical_and(qp_grid[:,np.newaxis] < eps[np.newaxis, :], eps.reshape((1, -1)) < e_fermi)
            # This is still incorrect but should be a bit better
            mask_vir = np.logical_and(qp_grid[:,np.newaxis] > eps[np.newaxis, :], eps.reshape((1, -1)) > e_fermi)
            mask_occ = np.logical_and(qp_grid[:,np.newaxis] < eps[np.newaxis, :], eps.reshape((1, -1)) < e_fermi)
            fill_factors[mask_vir] = 1.
            fill_factors[mask_occ] = -1.

            # Treat a special case
            mask_vir_eq = np.logical_and(qp_grid[:,np.newaxis] == eps[np.newaxis, :], eps.reshape((1, -1)) > e_fermi)
            mask_occ_eq = np.logical_and(qp_grid[:,np.newaxis] == eps[np.newaxis, :], eps.reshape((1, -1)) < e_fermi)

            fill_factors[mask_vir_eq] = 0.5
            fill_factors[mask_occ_eq] = -0.5

            if self.low_mem: # Low memory algorithm; Calculates W  matrix elements only for those orbitals that have non-zero fill factors

                # This implementation will be a lot slower but will utilize much less memory
                # Basis set size is usually much smaller than the grid size for my systems,
                # yet if I compute the residue term for each basis function separately => this may reduce memory cost 20 times!
                for m in range(nbf):
                    g_index = np.arange(ngrid)
                    g_res = g_index[fill_factors[:, m] != 0.]
                    if len(g_res) == 0:
                        continue
                    else:
                        #print(len(g_res))
                        pass

                    if self.analytic_W:

                        Ds_p__ = offset_complex_eps[m, g_res, np.newaxis] + self.omega_s[np.newaxis,:] - 0.5j*self.eta
                        Ds_m__ = offset_complex_eps[m, g_res, np.newaxis] - self.omega_s[np.newaxis,:] + 0.5j*self.eta

                        assert Ds_p__.shape == Ds_m__.shape and Ds_p__.shape == (len(g_res), len(self.omega_s))
                        Wnm_4res[m, g_res] = np.einsum("s, s, gs  -> g",omega_rts[nocc - no_qp + qp, m,:], omega_rts[nocc - no_qp + qp,m,:], 1./Ds_m__ - 1./Ds_p__)

                    else:

                        O__ = self.nmR[nocc-no_qp + qp, m, :]
                        dp__ = offset_complex_eps[m, g_res, np.newaxis, np.newaxis] + e_ai[np.newaxis, :, :] - 0.501j * self.eta
                        dm__ = offset_complex_eps[m, g_res, np.newaxis, np.newaxis] - e_ai[np.newaxis, :, :] + 0.501j * self.eta
                        zero_thresh = 1e-12
                        assert np.min(np.abs(dp__)) > zero_thresh and np.min(np.abs(dm__)) > zero_thresh
                        id_pq = np.eye(naux)
                        #Ppq__ = np.einsum("aiP, gai, aiQ->gPQ", self.nmR[nocc:, :nocc,:], 1./dm__ - 1./dp__, self.nmR[nocc:,:nocc,:])
                        Ppq__ = 2. * np.einsum("aiP, gai, aiQ->gPQ", self.nmR[nocc:, :nocc,:], 1./dm__ - 1./dp__, self.nmR[nocc:,:nocc,:])
                        del dp__
                        del dm__
                        Wnm_4res[m, g_res] = np.einsum("P, gPQ, Q -> g", O__, (np.linalg.inv(id_pq[np.newaxis, :,:] - Ppq__) - id_pq[np.newaxis, :,:]), O__)
                        if self.debug:
                            # loop over the grid and check if matrix inverse was performed with sufficient accuracy
                            thresh = 1e-11
                            for g in range(len(g_res)):
                                tmp = id_pq - Ppq__[g, :, :]
                                tmp_1 = np.linalg.inv(tmp)
                                diff = np.dot(tmp, tmp_1) - id_pq
                                if not (np.max(np.real(diff)) < thresh and np.max(np.imag(diff)) < thresh):
                                    print("Matrix inverse cannot be performed with sufficient accuracy!")
                                    print("Errors (real and imaginary parts)")
                                    print(np.max(np.real(diff)))
                                    print(np.max(np.imag(diff)))
                                    assert np.max(np.real(diff)) < thresh and np.max(np.imag(diff)) < thresh



            else: # Simple algorithm; supposed to be fast (especially for analytic W) but not memory eifficient

                if self.analytic_W:

                    Ds_p__ = offset_complex_eps[:,:,np.newaxis] + self.omega_s[np.newaxis, np.newaxis,:] - 0.5j*self.eta
                    Ds_m__ = offset_complex_eps[:,:,np.newaxis] - self.omega_s[np.newaxis, np.newaxis,:] + 0.5j*self.eta

                    assert Ds_p__.shape == Ds_m__.shape and Ds_p__.shape == (nbf, ngrid, len(self.omega_s))
                    Wnm_4res = np.einsum("ms, ms, mgs  -> mg",omega_rts[nocc - no_qp + qp,:,:], omega_rts[nocc - no_qp + qp,:,:], 1./Ds_m__ - 1./Ds_p__)

                else:

                    O__ = self.nmR[nocc-no_qp + qp, :, :]
                    dp__ = offset_complex_eps[:, :, np.newaxis, np.newaxis] + e_ai[np.newaxis, np.newaxis, :, :] - 0.501j * self.eta
                    dm__ = offset_complex_eps[:, :, np.newaxis, np.newaxis] - e_ai[np.newaxis, np.newaxis, :, :] + 0.501j * self.eta

                    zero_thresh = 1e-10
                    assert np.min(np.abs(dp__)) > zero_thresh and np.min(np.abs(dm__)) > zero_thresh
                    id_pq = np.eye(naux)
                    assert id_pq.shape == (naux, naux) and np.all(np.diag(id_pq) == np.ones(naux))
                    #Ppq__ = np.einsum("aiP, mgai, aiQ->mgPQ", self.nmR[nocc:, :nocc,:], 1./dm__ - 1./dp__, self.nmR[nocc:,:nocc,:])
                    Ppq__ = 2. * np.einsum("aiP, mgai, aiQ->mgPQ", self.nmR[nocc:, :nocc,:], 1./dm__ - 1./dp__, self.nmR[nocc:,:nocc,:])
                    del dp__
                    del dm__

                    assert Ppq__.shape == (nbf, ngrid, naux, naux)
                    assert O__.shape == (nbf, naux)
                    Wnm_4res = np.einsum("mP, mgPQ, mQ -> mg", O__, (np.linalg.inv(id_pq[np.newaxis, np.newaxis, :,:] - Ppq__) - id_pq[np.newaxis, np.newaxis, :,:]), O__)



            R_term = np.einsum("gm, mg->g", fill_factors, Wnm_4res)
            Rn[qp, :] = np.copy(R_term)

            #quick sanity check for the residue term
            within_gap = np.logical_and(qp_grid < eps[nocc], qp_grid > eps[nocc - 1])
            if self.debug:
                print("R term within the HOMO-LUMO gap ", R_term[within_gap])

        return Rn - In

