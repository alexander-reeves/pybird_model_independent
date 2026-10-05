from pybird.module import * 
import numpy as np
import jax


class Cosmo():
    """A Class to compute the linear power spectrum and growth factors from cosmological parameters.
    
    This class provides an interface to various linear cosmology computation engines,
    including CLASS, CosmoPower emulator, and symbolic solvers. It handles
    the calculation of linear matter power spectra, scale-independent growth factors and rates,
    and other cosmological quantities needed for perturbation theory calculations.
    
    Attributes:
        c (dict): Configuration dictionary holding parameters for calculations.
        
    Methods:
        set_cosmo(): Compute linear cosmological quantities from input parameters using 
            a specified backend module (class, CPJ, or Symbolic). Returns a dictionary containing 
            computed quantities like the linear matter power spectrum, growth factors and rates,
            and parameters for AP effect.
    """ 
    def __init__(self, config):
        self.c = config

    def _is(self, module, module_name): 
        return module.casefold() == module_name.casefold()

    def set_cosmo(self, cosmo_dict, module='class', engine=None):
        
        # Handle None cosmo_dict
        if cosmo_dict is None:
            cosmo_dict = {}

        # checking redshift
        if self.c["z"] is None:
            if "z" not in cosmo_dict: raise Exception("Please provide a 'z' in cosmo_dict or when better, when providing options to correlator() through .set({\'z\': z, ...})")
            else:
                self.c["z"] = cosmo_dict["z"]
                cosmo_dict.pop("z")
        else: 
            if "z" in cosmo_dict: 
                if cosmo_dict["z"] != self.c["z"]: raise Exception("The provided z in cosmo_dict is different than the one set in correlator?")
                else: cosmo_dict.pop("z")

        cosmo = {}

        log10kmax = 0.
        #if self.c["with_nnlo_counterterm"]: log10kmax = 1 # slower, but required for the wiggle-no-wiggle split scheme
        cosmo["kk"] = logspace(-5, log10kmax, 512)  # k in h/Mpc

        if self._is(module, 'class') or self._is(module, 'classy'):

            if not engine:
                from classy import Class
                cosmo_dict_local = cosmo_dict.copy()
                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                else: 
                    zmax = self.c["z"]
                M = Class()
                M.set(cosmo_dict_local)
                M.set({'output': 'mPk', 'P_k_max_h/Mpc': 10.**log10kmax, 'z_max_pk': zmax, })
                #     'tol_perturbations_integration': 1.e-6, 'tol_background_integration': 1.e-5, 'k_per_decade_for_pk': 200, 'k_per_decade_for_bao': 200})
                M.compute()
            else: M = engine

            cosmo["pk_lin"] = array([M.pk_lin(k*M.h(), self.c["z"])*M.h()**3 for k in cosmo["kk"]]) # P(k) in (Mpc/h)**3

            if self.c["multipole"] > 0: 
                cosmo["f"] = M.scale_independent_growth_factor_f(self.c["z"])
            if not self.c["with_time"]:
                cosmo["D"] = M.scale_independent_growth_factor(self.c["z"])
            if self.c["with_nonequal_time"]:
                cosmo["D1"] = M.scale_independent_growth_factor(self.c["z1"])
                cosmo["D2"] = M.scale_independent_growth_factor(self.c["z2"])
                cosmo["f1"] = M.scale_independent_growth_factor_f(self.c["z1"])
                cosmo["f2"] = M.scale_independent_growth_factor_f(self.c["z2"])
            if self.c["with_exact_time"] or self.c["with_quintessence"]:
                cosmo["z"] = self.c["z"]
                cosmo["Omega0_m"] = M.Omega0_m()
            # if "w0_fld" in cosmo_dict:
            #     cosmo["w0_fld"] = cosmo_dict["w0_fld"]
            if self.c["with_ap"]:
                cosmo["H"], cosmo["DA"] = M.Hubble(self.c["z"]) / M.Hubble(0.), M.angular_distance(self.c["z"]) * M.Hubble(0.)

            if self.c["with_redshift_bin"]:
                def comoving_distance(z): return M.angular_distance(z) * (1+z) * M.h()
                cosmo["Dz"] = array([M.scale_independent_growth_factor(z) for z in self.c["redshift_bin_zz"]])
                cosmo["fz"] = array([M.scale_independent_growth_factor_f(z) for z in self.c["redshift_bin_zz"]])
                cosmo["rz"] = array([comoving_distance(z) for z in self.c["redshift_bin_zz"]])

            if self.c["with_quintessence"]:
                # starting deep inside matter domination and evolving to the total adiabatic linear power spectrum.
                # This does not work in the general case, e.g. with massive neutrinos (okish for minimal mass though)
                # This does not work for 'with_redshift_bin': True. # eventually to code up
                zm = 5. # z in matter domination
                def scale_factor(z): return 1/(1.+z)
                Omega0_m = cosmo["Omega0_m"]
                w = cosmo["w0_fld"]
                GF = GreenFunction(Omega0_m, w=w, quintessence=True)
                Dq = GF.D(scale_factor(zfid)) / GF.D(scale_factor(zm))
                Dm = M.scale_independent_growth_factor(self.c["z"]) / M.scale_independent_growth_factor(zm)
                cosmo["pk_lin"] *= Dq**2 / Dm**2 * ( 1 + (1+w)/(1.-3*w) * (1-Omega0_m)/Omega0_m * (1+zm)**(3*w) )**2 # 1611.07966 eq. (4.15)
                cosmo["f"] = GF.fplus(1/(1.+self.c["z"]))

            # wiggle-no-wiggle split # algo: 1003.3999; details: 2004.10607
            def get_smooth_wiggle_resc(kk, pk, alpha_rs=1.): # k [h/Mpc], pk [(Mpc/h)**3]
                kp = linspace(1.e-7, 7, 2**16)   # 1/Mpc
                ilogpk = interp1d(log(kk * M.h()), log(pk / M.h()**3), fill_value="extrapolate") # Mpc**3
                lnkpk = log(kp) + ilogpk(log(kp))
                harmonics = dst(lnkpk, type=2, norm='ortho')
                odd, even = harmonics[::2], harmonics[1::2]
                nn = arange(0, odd.shape[0], 1)
                nobao = delete(nn, arange(120, 240,1))
                smooth_odd = interp1d(nn, odd, kind='cubic')(nobao)
                smooth_even = interp1d(nn, even, kind='cubic')(nobao)
                smooth_odd = interp1d(nobao, smooth_odd, kind='cubic')(nn)
                smooth_even = interp1d(nobao, smooth_even, kind='cubic')(nn)
                smooth_harmonics =  array([[o, e] for (o, e) in zip(smooth_odd, smooth_even)]).reshape(-1)
                smooth_lnkpk = dst(smooth_harmonics, type=3, norm='ortho')
                smooth_pk = exp(smooth_lnkpk) / kp
                wiggle_pk = exp(ilogpk(log(kp))) - smooth_pk
                spk = interp1d(kp, smooth_pk, bounds_error=False)(kk * M.h()) * M.h()**3 # (Mpc/h)**3
                wpk_resc = interp1d(kp, wiggle_pk, bounds_error=False)(alpha_rs * kk * M.h()) * M.h()**3 # (Mpc/h)**3 # wiggle rescaling
                kmask = where(kk < 1.02)[0]
                return kk[kmask], spk[kmask], pk[kmask] #spk[kmask]+wpk_resc[kmask]

            #if self.c["with_nnlo_counterterm"]: cosmo["kk"], cosmo["Psmooth"], cosmo["pk_lin"] = get_smooth_wiggle_resc(cosmo["kk"], cosmo["pk_lin"])

            return cosmo

        elif self._is(module, 'Symbolic'):

            if not engine: 
                from pybird.symbolic import Symbolic
                M = Symbolic(); M.set(cosmo_dict)
            else: 
                M = engine
            
            M.compute(cosmo["kk"], self.c['z'])
            
            cosmo['pk_lin'] = M.pk_lin
            cosmo['D'], cosmo['f'] = M.D, M.f
            if self.c["with_ap"]: cosmo['H'], cosmo['DA'] = M.H, M.DA

            return cosmo

        elif self._is(module, 'CPJ'):

            def to_Mpc_per_h_jax(_pk, _kk, h):
                ilogpk_ = interp1d(log(_kk), log(_pk), fill_value='extrapolate')
                return exp(ilogpk_(log(_kk*h))) * h**3

            if not engine:
                from pybird.integrated_model_jax import IntegratedModel
                from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
                cosmo_dict_local = cosmo_dict.copy()

                M = CPJ(probe='mpk_lin')
                M_growth = IntegratedModel(None, None, None)
                M_growth.restore(self.c["emu_path"] + "/growth_model.h5") 

                if self.c["with_bias"] and "bias" in cosmo_dict: del cosmo_dict_local["bias"] # remove to not pass it to classy that otherwise complains
                # if not self.c["with_time"] and "A" in cosmo_dict: del cosmo_dict_local["A"] # same as above
                # if self.c["with_redshift_bin"]: zmax = max(self.c["redshift_bin_zz"])
                # else: zmax = self.c["z"]
                zmax = self.c["z"]  

            else:
                M, M_growth, cosmo_dict_local = engine.CPJ, engine.growth, engine.cosmo

            try: 
                input_dict_pk = {key: array([cosmo_dict_local[key]]) for key in ["omega_b", "omega_cdm", "n_s", "ln10^{10}A_s", "h"]}
            
            except Exception(e):
                print("the input dict did not build... probably you are missing some of the required cosmo inputs for the emu")
                print("exception:", e) 
            
            from pybird.symbolic import DA, Hubble, f, D as D_sym

            Omega_m = (cosmo_dict_local["omega_cdm"] + cosmo_dict_local["omega_b"]) / cosmo_dict_local["h"]**2
            h_loc = cosmo_dict_local["h"]

            # Optional reference-redshift evaluation: P(k, z) = P(k, z_ref) D(z)^2 / D(z_ref)^2 (symbolic LCDM growth)
            z_ref = self.c["cpj_z_ref"] if "cpj_z_ref" in self.c else -1.
            if z_ref is not None and z_ref >= 0.:
                input_dict_pk["z"] = array([z_ref])
                growth_resc = (D_sym(Omega_m, self.c["z"]) / D_sym(Omega_m, z_ref))**2
            else:
                input_dict_pk["z"] = array([self.c["z"]])
                growth_resc = 1.

            pk_mpc = M.predict(input_dict_pk)  # Mpc^3 on M.modes [1/Mpc]

            # Optional evaluation on the emulator knots: P_h(k_h) = P_Mpc(k_h h) h^3 at k_h = knots
            if "cpj_pk_on_knots" in self.c and self.c["cpj_pk_on_knots"]:
                knots = array(load(self.c["knots_path"]))  # h/Mpc
                # linear log-log interpolation from the dense CosmoPower grid (same as jnp.interp)
                cosmo["pk_lin"] = growth_resc * exp(interp(log(knots * h_loc), log(array(M.modes)), log(pk_mpc))) * h_loc**3
                cosmo["kk"] = knots
            else:
                cosmo["pk_lin"] = growth_resc * array(to_Mpc_per_h_jax(pk_mpc, M.modes, h_loc))
                cosmo["kk"] = array(M.modes)

            ### using LCDM growths and distances for now (until growth emulator is debugged)

            cosmo["f"] = f(Omega_m, self.c["z"])
            cosmo["H"], cosmo["DA"] = Hubble(Omega_m, self.c["z"]), DA(Omega_m, self.c["z"])

            # if self.c["multipole"] > 0: 
            #     sigma_8s = array([0.8]) # this has no impact on growth- I mistakenly included in training so this is a dummy value
            #     emulator_growth_input = stack([
            #         input_dict_pk["omega_b"],
            #         input_dict_pk["omega_cdm"],
            #         input_dict_pk["n_s"],
            #         sigma_8s,
            #         input_dict_pk["h"],
            #         array([self.c["z"]])
            #     ], axis=1)

            #     D, f, H, DA = M_growth.predict(emulator_growth_input)[0]
            #     cosmo["f"] = f
            # if not self.c["with_time"]: cosmo["D"] = D
            # if self.c["with_ap"]: cosmo["H"], cosmo["DA"] = H, DA

            
        elif self._is(module, 'CPJ_custom'):

            # cd cosmopower_jax/cosmopower_jax/trained_models; git clone https://github.com/cosmopower-organization/mnu.git
            # might need to first: pip install tensorflow
            # make sure that you have CPJ in editable mode: pip install -e .  

            from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
            
            def to_Mpc_per_h_jax(_pk, _kk, h):
                ilogpk_ = interp1d(log(_kk), log(_pk), fill_value='extrapolate')
                return exp(ilogpk_(log(_kk*h))) * h**3

            def get_pk_lin_from_cpj_custom(cosmo, kk, z):
                _cosmo = {key: array([cosmo[key]]) for key in ["omega_b", "omega_cdm", "n_s", "ln10^{10}A_s", "m_ncdm"]}
                _cosmo['H0'] = array([cosmo['h'] * 100.])
                _cosmo['z_pk_save_nonclass'] = array([z])
                
                M = CPJ(probe='custom', filename=os.path.join('mnu', 'PK', 'PKL_mnu_v1.npz'))
                pk = M.predict(_cosmo)
                ndspl = 10
                k_arr = np.geomspace(1e-4,50.,5000)[::ndspl]
                ls = np.arange(2,5000+2)[::ndspl]
                dls = ls*(ls+1.)/2./np.pi
                pk = 10.**array(pk)
                pk =  ((dls)**-1*pk)
                pk = array(to_Mpc_per_h_jax(pk, k_arr, cosmo['h']))
                ipk = interp1d(log(k_arr), log(pk), kind='linear', fill_value='extrapolate')
                pk = exp(ipk(log(kk)))
                return pk
            
            def get_growth(cosmo, z):
                _cosmo = {key: array([cosmo[key]]) for key in ["omega_b", "omega_cdm", "n_s", "ln10^{10}A_s", "m_ncdm"]}
                _cosmo['H0'] = array([cosmo['h'] * 100.])

                z_arr = linspace(0., 20., 5000)
                dz = z_arr[-1] - z_arr[-2]
                idx = abs(z_arr - z).argmin()

                M = CPJ(probe='custom_log', filename=os.path.join('mnu', 'growth-and-distances', 'HZ_mnu_v1.npz'))
                Hz = M.predict(_cosmo)
                H = Hz[idx] / Hz[0]
                
                M = CPJ(probe='custom', filename=os.path.join('mnu', 'growth-and-distances', 'DAZ_mnu_v1.pkl'))
                DAz = M.predict(_cosmo)
                DA = DAz[idx] * Hz[0]

                M = CPJ(probe='custom', filename=os.path.join('mnu', 'growth-and-distances', 'S8Z_mnu_v1.npz'))
                s8z = M.predict(_cosmo)
                
                fs8z = - (s8z[idx+1]-s8z[idx-1]) / (2.*dz) * (1.+z) # f sigma8 = d sigma8/d ln a = - (d sigma8/dz)*(1+z)
                f = fs8z / s8z[idx]
                
                return H, DA, f

            cosmo['kk'] = geomspace(2e-4, 1, 500)
            kk, z = cosmo['kk'], self.c['z']
            cosmo['pk_lin'] = get_pk_lin_from_cpj_custom(engine.cosmo, kk, z)
            H, DA, f = get_growth(engine.cosmo, z)
            cosmo["H"], cosmo["DA"], cosmo["f"] = H, DA, f

        elif self._is(module, 'IEmu'):
            # Internal CLASS GR+w0wa P_lin emulator (Flax export of UPanda-trained model).
            # Ported from pybird_emu commit 05a92bd. Emulates the FULL w0wa linear P(k)
            # directly over a wide box (w0[-2.1,0.4], wa[-3.6,1.0], ...); growth/AP from
            # the Symbolic analytic helpers. The training log-preprocess is inverted HERE
            # (not inside IntegratedModel.predict) so other emulators are unaffected.
            from pybird.symbolic import D as D_sym, f as f_sym, Hubble as H_sym, DA as DA_sym

            if not engine:
                import os
                from pybird.integrated_model_jax import IntegratedModel
                cosmo_dict_local = cosmo_dict.copy()
                pklin_h5 = self.c['iemu_pklin_path'] if 'iemu_pklin_path' in self.c else None
                if not pklin_h5:
                    emu_path = self.c['emu_path'] if 'emu_path' in self.c else ''
                    cand = os.path.join(emu_path, 'pklin_gr_w0wa_class_jax_model.h5')
                    if emu_path and os.path.exists(cand):
                        pklin_h5 = cand
                    else:
                        pklin_h5 = os.path.join(os.path.dirname(__file__), '..', 'data', 'emu', 'pklin_gr_w0wa_class_jax_model.h5')
                M_pk = IntegratedModel(None, None, None)
                M_pk.restore(pklin_h5)
                if hasattr(M_pk, 'modes'):
                    modes = array(M_pk.modes)
                else:
                    k_file = os.path.join(os.path.dirname(pklin_h5), 'pklin_gr_w0wa_class_k.npy')
                    modes = array(load(k_file)) if os.path.exists(k_file) else logspace(-5, 0, 512)
            else:
                M_pk, cosmo_dict_local, modes = engine.pklin, engine.cosmo, engine.modes

            c = cosmo_dict_local
            h = c['h']
            m_nu = c['m_ncdm'] if 'm_ncdm' in c else (c['m_nu'] if 'm_nu' in c else 0.)
            w0 = c['w0_fld'] if 'w0_fld' in c else (c['w0'] if 'w0' in c else -1.)
            wa = c['wa_fld'] if 'wa_fld' in c else (c['wa'] if 'wa' in c else 0.)
            if 'Omega_b' in c:
                Omega_b = c['Omega_b']
            else:
                Omega_b = c['omega_b'] / h**2
            if 'Omega_m' in c:
                Omega_m = c['Omega_m']
            else:
                Omega_m = (c['omega_cdm'] + c['omega_b'] + m_nu / 93.14) / h**2
            if 'A_s' in c:
                A_s_1e9 = c['A_s'] * 1e9
            else:
                A_s_1e9 = exp(c['ln10^{10}A_s']) / 10.

            z = self.c['z']
            # UPanda training order: A_s_1e9, Omega_m, Omega_b, h, n_s, m_nu, w0, wa, a
            x = stack([
                array([A_s_1e9]),
                array([Omega_m]),
                array([Omega_b]),
                array([h]),
                array([c['n_s']]),
                array([m_nu]),
                array([w0]),
                array([wa]),
                array([1. / (1. + z)]),
            ], axis=1)

            pk_pred = M_pk.predict(x)[0]
            if getattr(M_pk, 'log_preprocess', False):
                pk_pred = exp(pk_pred) + 2. * M_pk.offset   # invert training log-preprocess -> P(k) [(Mpc/h)^3]
            cosmo['pk_lin'] = array(pk_pred)
            cosmo['kk'] = array(modes)

            cosmo['D'], cosmo['f'] = D_sym(Omega_m, z, w0, wa), f_sym(Omega_m, z, w0, wa)
            if self.c["with_ap"]:
                cosmo['H'], cosmo['DA'] = H_sym(Omega_m, z, w0, wa), DA_sym(Omega_m, z, w0, wa)

            return cosmo

        elif module is None:
            # no cosmo module -assume you have already input your required pk_lin 
            cosmo = cosmo_dict

        return cosmo
