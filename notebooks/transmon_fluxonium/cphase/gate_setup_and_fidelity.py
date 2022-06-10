#!/usr/bin/env python
# coding: utf-8

# # Ficheux-like gate between a transmon and a fluxonium
# 
# In this notebook we study a two-qubit gate between a transmon and a fluxonium inspired by the two-qubit gate between a transmon and a fluxonium described in Ficheux et al. Phys. Rev. X 11, 021026 (2021)

# In[24]:


import numpy as np
import time
import qutip as qtp
import cmath
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pysqkit
from pysqkit.util.linalg import get_mat_elem, tensor_prod,     hilbert_schmidt_prod
from pysqkit.util.metrics import average_process_fidelity,     average_gate_fidelity
from pysqkit.util.phys import temperature_to_thermalenergy
import pysqkit.util.transformations as trf
from typing import List, Dict, Callable
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

from IPython.display import display, Latex

#%matplotlib notebook


# The gate is based on the following driven Hamiltonian of a fluxonium and a transmon capacitively coupled
# 
# $$H = H^{(0)}_T + H^{(0)}_F + V + H_{drive},$$
# 
# $$H^{(0)}_T =  \hbar \omega_T \hat{b}^{\dagger}\hat{b} + \hbar \frac{\delta_T}{2} \hat{b}^{\dagger} \hat{b}^{\dagger}\hat{b} \hat{b},$$
# 
# $$H^{(0)}_F = 4 E_{C,F} \cdot \hat{n}^2_F + \frac{1}{2}E_{L,F}\cdot\hat{\varphi}^2_F - E_{J,F}\cos\left(\hat{\varphi}_F - \phi_{ext,F}\right),$$
# 
# $$V = J_C \cdot \hat{n}_T \cdot \hat{n}_F,$$
# 
# $$\frac{H_{drive}}{h} = \left( \varepsilon_T \hat{n}_T + \varepsilon_F \hat{n}_F \right) \cos(2\pi f_d t).$$
# 
# The (approximate) charge operator for the transmon in terms of annihilation and creation operators reads
# 
# $$
# \hat{n}_T = i \biggl( \frac{E_J}{32 |\delta_T |} \biggr)^{1/4} (b^{\dagger} - b),
# $$
# where 
# $$
# E_J = \hbar \biggl(\frac{\omega_T - \delta_T}{8 | \delta_T |} \biggr)^2.
# $$
# 

# # Introducing the qubits and the coupled system

# In[25]:


temperature = 0.020 # K
thermal_energy = temperature_to_thermalenergy(temperature) # kb T/h in GHz
d_comp = 4

#Transmon
levels_t = 3
transm = pysqkit.qubits.SimpleTransmon(
    label='T', 
    max_freq=4.5, 
    anharm=-0.3,
    diel_loss_tan=0.75*1e-6, #set to zero to check d_1 L1 = d_2 L2
    env_thermal_energy=thermal_energy,    
    dim_hilbert=levels_t
)

#Fluxonium
levels_f = 5
ec_f = .973
el_f = .457
ej_f = 8.0
# The external flux in units of \phi_0 is set to 1/2 by defauls:
# double well configuration

flx = pysqkit.qubits.Fluxonium(
    label='F', 
    charge_energy=.973, 
    induct_energy=.457, 
    joseph_energy=8.0, 
    diel_loss_tan=7.1*1e-6*0, #set to zero to check d_1 L1 = d_2 L2
    env_thermal_energy=thermal_energy
)
flx.diagonalize_basis(levels_f)

# We also add a drive on the fluxonium
flx.add_drive(
    pysqkit.drives.microwave_drive,
    label='cz_drive_f',
    pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
)

d_leak = levels_t*levels_f - d_comp

jc = 0.07
coupled_sys = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=jc)
bare_system = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=0.0)


# In[26]:


def energy_levels_diagram(
    system: pysqkit.systems.system.QubitSystem,
    levels_to_plot: List[str],
    plot_setup={'fs': 20, 'lw': 2.0, 'lw_levels': 3.0, 'ls': 16},
    detuning_tol=3.0,
    show_drive=True
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ground_energy = system.state('00')[0]
    energy = {}
    state = {}
    step = 11.0
    level_width = 5.0
    x_avg = {}
    
    for level in levels_to_plot:
        energy[level] = system.state(level)[0] - ground_energy
        state[level] = system.state(level)[1]
    for level in levels_to_plot:
        level_as_list = [int(lev) for lev in level]
        x_min = -0.5 + step*level_as_list[1] - step*level_as_list[0]
        x_max = x_min + level_width
        x_avg[level] = (x_min + x_max)/2
        ax.hlines(y=energy[level], xmin=x_min, xmax=x_max, linewidth=plot_setup['lw_levels'], color='k')
        ax.text(x_max+0.2, energy[level]-0.05, r'$ \vert' + level + ' \\rangle$', 
                fontsize=plot_setup['fs'], zorder=10)
        ax.tick_params(labelsize=plot_setup['ls'])
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    qubit_labels = system.labels
    n1n2_op = system[qubit_labels[0]].charge_op().dot(coupled_sys[qubit_labels[1]].charge_op())
    abs_mat_elem_vec = np.zeros(len(levels_to_plot)**2)
    count = 0
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem_vec[count] = np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            count += 1
    max_abs_mat_elem = np.max(abs_mat_elem_vec)
    for level_1 in levels_to_plot:
        for level_2 in levels_to_plot:
            abs_mat_elem = np.abs(get_mat_elem(n1n2_op, state[level_1], state[level_2]))
            shade_factor = abs_mat_elem/max_abs_mat_elem
            if shade_factor > 1e-4 and np.abs(energy[level_1] - energy[level_2]) < detuning_tol:
                ax.annotate('', (x_avg[level_1], energy[level_1]), (x_avg[level_2], energy[level_2]), 
                            arrowprops=dict(color='darkorange', alpha=shade_factor**5,\
                                            lw=plot_setup['lw'], arrowstyle='<->'))
                
    if show_drive:
        ax.annotate('', (x_avg['10'], energy['10']), (x_avg['13'], energy['13']), 
                            arrowprops=dict(color='darkblue', lw=plot_setup['lw'],\
                                            arrowstyle='<->', linestyle = '--'))
        ax.annotate('', (x_avg['00'], energy['00']), (x_avg['03'], energy['03']), 
                            arrowprops=dict(color='darkblue', lw=plot_setup['lw'],\
                                            arrowstyle='<->', linestyle = '--'))
        
    ax.set_title(r'$\mathrm{Transmon} \leftarrow \quad \rightarrow  \mathrm{Fluxonium}$',                  {'fontsize': plot_setup['fs']})
    ax.set_ylabel(r'$\mathrm{Energy} \, (\mathrm{GHz})$', fontsize=plot_setup['fs'])
    plt.show()    


# The basis principle behind the gate is illustrated in the following figure

# In[27]:


level_list = ['00', '01', '10', '11', '02', '20', '12','21', '03', '13', '04']

energy_levels_diagram(bare_system, level_list)


# The figure shows the energy of the bare levels. The orange arrows indicate the non-zero matrix elements between the (bare) levels provided by the capacitive coupling $V$: the more shaded the arrow the weaker the coupling. The idea is to design the system so that the $| 1 3 \rangle$-$|0 4 \rangle$ is close to resonance so that the levels get sufficiently hybridized. A CZ gate can then be realized by driving the $|0 \rangle$-$|3 \rangle$ transition of the fluxonium, which induces transitions depicted by the blue, dashed arrow in the figure. The scheme is inspired by the one between two fluxonia described in Ficheux et al. Phys. Rev. X 11, 021026 (2021), where the $|2 1  \rangle$ and $|12 \rangle$ levels are hybridized and the transition $| 1 \rangle$-$|2 \rangle$ of one of the two fluxonia is driven.
# 
# We look at the following quantities : 
# 
# $$\zeta_{ZZ} = E_{00} + E_{11} - E_{01} - E_{10}$$
# $$\Delta = \left( E_{13} - E_{10} \right) -  \left( E_{03} - E_{00} \right)$$
# 
# The $\Delta$ is similar at the one in Ficheux but looking at the 0-3 transition of the Fluxonium.

# In[28]:


def zz(
    system: pysqkit.systems.system.QubitSystem
) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0]     - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def delta(
    system: pysqkit.systems.system.QubitSystem
) -> float:
    delta_gate = (system.state('13')[0] - system.state('10')[0]) -     (system.state('03')[0] - system.state('00')[0])
    return delta_gate


# For the given parameters we have

# In[29]:


display(Latex(r'$\xi_{{ZZ}}/h = {:.3f} \, \mathrm{{MHz}}$'.format(1e3*zz(coupled_sys))))
display(Latex(r'$\Delta/h = {:.3f} \, \mathrm{{MHz}}$'.format(1e3*delta(coupled_sys))))


# ## Matching the parameters

# In[30]:


def generalized_rabi_frequency(
    levels: List['str'],
    eps: Dict,
    drive_frequency: float,
    system: pysqkit.systems.system.QubitSystem
) -> float:
    if len(levels) != 2:
        raise ValueError('The generalized rabi frequency is defined betwen two levels. '
                        'Please specify the desired two levels.')
        
    qubit_labels = system.labels
    drive_op = 0
    for label in qubit_labels:
        drive_op += eps[label]*system[label].charge_op()
    
    in_energy, in_state = system.state(levels[0])
    out_energy, out_state = system.state(levels[1])   
    big_omega_transition = np.abs(get_mat_elem(drive_op, in_state, out_state))
    
    energy_transition = np.abs(in_energy - out_energy)
    drive_detuning = energy_transition - drive_frequency
    
    return np.sqrt(big_omega_transition**2 + drive_detuning**2)


# For this case we look at the 10-13 and 00-03 transitions so we process the corresponding Rabi frequencies :
# 
# $$\Omega_{10-13}  = \left| \epsilon_T \langle 10 | \hat{n}_T | 13 \rangle + \epsilon_F \langle 10 | \hat{n}_F | 13 \rangle \right|$$
# $$\Omega_{00-03}  = \left| \epsilon_T \langle 00 | \hat{n}_T | 03 \rangle + \epsilon_F \langle 00 | \hat{n}_F | 03 \rangle \right|$$
# 
# And then :
# $$r = \dfrac{\Omega_{10-13}}{\Omega_{00-03}} \text{  and  } \dfrac{\delta}{\Delta} = \dfrac{r^2 - \sqrt{(r^2-1)^2 + r^2 }} {r^2 - 1}$$ 
# 
# With these conventions taken in the paper :
# $$\Delta = f_{10-13} - f_{00-03}$$
# $$\delta = f_{10-13} - f_d$$
# 
# These conventions are taken in orde to have : $\Omega = \sqrt{\Omega_{10-13}^2  + \delta^2} = \sqrt{\Omega_{00-03}^2  + (\delta - \Delta)^2}$
# 
# The $\Delta$ is fixed by the parameters of the qubits, we now need to settle the values for the drive parameters.
# 
# First we need to fix the $\varepsilon$.
# 
# We take the condition of the paper :
# $$\Omega = \Delta$$
# 
# However, one equation for two parameters is not enough. We assume that the fluxonium is driven much more than the transmon (as we are mainly looking at the 0-3 transtion of F) and, in general, we could take the drive on the transmon to be a fixed fraction of the one on the fluxonium. In what follows we simply set the drive on the transmon to be zero. Now we have enough constraints to find values for every parameters.
# 
# 
# With this $\delta$, we have $\Omega = \Delta$ and a CZ gate is achieved with zero leakage in time exactly $t_{drive} = 1/\Delta$.
# 
# In order to setup the optimization define the following function that respects the scipy.optimize.minimize syntax. The vector of variable x0 in this case has length 2: the first entry is a reference drive strength, while the second the drive frequency. The drive parameters for each qubit are defined by providing the ratio between the qubit drive strength and the reference drive strength. This can be used to take into account potential classical crosstalk.

# In[31]:


def func_to_minimize(
    x0: np.ndarray,
    levels_first_transition: List['str'],
    levels_second_transition: List['str'],
    system: pysqkit.systems.system.QubitSystem,
    eps_ratio_dict: Dict    
) -> float:
    
    """
    Function to minimize in order to match the parameters in order to 
    implement a CZ gate up to single-qubit rotations. It returns the modulus
    of [rabi_second_transition - rabi_first_transition, 
    delta_gate - rabi_first_transition]/delta_gate
    x0 : np.ndarray([eps_reference, drive_freq]) represents the parameters to be 
         minimized.
    levels_first_transition : List with the labels of the first transition whose
       generalized Rabi frequency has to be matched
    levels_second_transition : List with the labels of the second transition whose
                               generalized Rabi frequency has to be matched
    system: coupled system we are analyzing
    eps_ratio_dict: dictionary whose keys are system.labels. The entries correspond
                    to the ratios between the corresponding qubit drive and the 
                    reference drive.     
    
    """
    
    qubit_labels = system.labels
    eps = {}
    for qubit in qubit_labels:
        eps[qubit] = x0[0]*eps_ratio_dict[qubit]
    rabi_first_transition = generalized_rabi_frequency(levels_first_transition, eps, x0[1], system)
    rabi_second_transition = generalized_rabi_frequency(levels_second_transition, eps, x0[1], system)
    delta_gate = delta(system)
    y = np.sqrt( (rabi_first_transition - rabi_second_transition)**2 +                 (rabi_first_transition - delta_gate)**2)
    return np.abs(y/delta_gate)


# In[32]:


x0 = np.array([0.03, 7.15]) #initial guess
eps_ratios = {'T': 0.0, 'F':1.0}
args_to_pass = (['00', '03'], ['10', '13'], coupled_sys, eps_ratios) 

start = time.time()

minimization_result = minimize(func_to_minimize, x0, args=args_to_pass)

end = time.time()

display(Latex(r'$\mathrm{{Minimization \, time}} = {:.3f} \, s$'.format(end - start)))
display(Latex(r'$f_{{\mathrm{{min}}}} = {:.2e}$'.format(minimization_result['fun'])))


# In[10]:


eps_drive = minimization_result['x'][0]
freq_drive = minimization_result['x'][1]
rabi_period = 1/delta(coupled_sys)
t_rise = 15.0 # [ns]
t_tot = 60    # [ns]
pts_per_drive_period = 5 

nb_points = int(t_tot*freq_drive*pts_per_drive_period)
tlist = np.linspace(0, t_tot, nb_points)

coupled_sys['F'].drives['cz_drive_f'].set_params(phase=0, time=tlist, rise_time=t_rise, 
                                                 amp=eps_drive, freq=freq_drive)


# We introduce the TomoEnv object. We also refine the absolute and relative tolerance of the qutip solver 

# In[11]:


simu_opt = qtp.solver.Options()
simu_opt.atol = 1e-12
simu_opt.rtol = 1e-10
env_syst = pysqkit.tomography.TomoEnv(system=coupled_sys, time=2*np.pi*tlist, options=simu_opt)


# The TomoEnv class can now return the superoperator representation of the quantum operation in an arbitrary Hilbert-Schmidt basis for a $d$-dimensional subspace, as long as this is a basis of diagonalizable operators. For instance the basis $|n \rangle \langle m|$, $n, m= 0, \dots, d-1$ is not a diagonalizable basis and thus the method cannot be used. 
# 
# In what follows we are interested in obtaining a superoperator representation of a quantum operation in the computational subspace that we here define

# In[12]:


state_labels = ['00', '01', '10', '11']
comp_states = []
for label in state_labels:
    comp_states.append(coupled_sys.state(label)[1])


# We remark that the specification of the subspace is left to the user and it has arbitrary dimension. The user must check that the given kets are indeed orthogonal. 
# 
# In what follows we will use the Weyl basis for qudits ($d>2$) as Hilbert-Schmidt basis. These are defined via the standard definition of $Z$ and $X$ operators for qudits (see Howard et al Nature volume 510, pages351–355 (2014)):
# 
# $$
# Z | j \rangle= \omega^j | j \rangle,
# $$
# $$
# X | j \rangle = | j + 1 \mod d \rangle,
# $$
# where $\omega = \exp \bigl[ \frac{2 \pi i}{d} \bigr ]$.
# 
# The (normalized) Weyl operators for qudits are defined as
# 
# $$
# D(\boldsymbol{\xi})=D \bigl( \boldsymbol{\xi} = \begin{bmatrix}\xi_x & \xi_z 
# \end{bmatrix}^T \bigr) = \frac{1}{\sqrt{d}} \omega^{- \frac{d+1}{2} \xi_x \xi_z} X^{\xi_x} Z^{\xi_z}, \quad \xi_x, \xi_z \in \mathbb{Z}_d = \{0, 1, \dots, d-1 \}.
# $$
# The Weyl operators form an orthonormal basis for the Hilbert space of (bounded) $d \times d$ matrices with Hilbert-Schmidt inner product
# 
# $$
# \langle A, B \rangle = \mathrm{Tr}(A^{\dagger} B).
# $$
# In fact
# $$
# \langle D(\boldsymbol{\xi}), D(\boldsymbol{\xi}') \rangle = \delta_{\boldsymbol{\xi}, \boldsymbol{\xi}'}.
#  $$
# 
# In order to use this basis to write our superoperator we simply need a function that orders them. Thus, this function should take as argument an index $i$ associated with the Weyl operator as well as the qudit dimension $d$ ($i < d^2$). The same is valid for any orthonormal Hilbert-Schmidt basis we wish to define, as long as it is a basis of diagonalizable matrices. 
# The following functions can also be found in pysqkit.util.hsbasis, but we here re-write them explicitly for illustration purposes.

# In[13]:


def weyl(
    xi_x: int,
    xi_z: int,
    d: int
) -> np.ndarray:

    """ 
    Returns the qudit Weyl operator with phase-space points xi_x, xi_z
    in a d x d phase space.
    We take the normalized version of the definition in 
    M. Howard et al, Nature 510, 351 (2014), i.e., divided by
    square root of d.
    """

    if xi_x >= d or xi_z >= d:
        raise ValueError("Qudit Pauli labels out of range.")

    if d <= 2:
        raise ValueError("The qudit dimension d must be larger than 2.")

    omega = np.exp(2*np.pi*1j/d)
    z = np.zeros([d, d], dtype=complex)
    x = np.zeros([d, d], dtype=complex)
    for k in range(0, d):
        z[k, k] = omega**k
        x[np.mod(k + 1, d), k] = 1
    z_pow = np.linalg.matrix_power(z, xi_z)
    x_pow = np.linalg.matrix_power(x, xi_x)

    return 1/np.sqrt(d)*omega**(-(d + 1)/2*xi_x*xi_z)*x_pow.dot(z_pow)

def weyl_by_index(
    i: int,
    d: int
) -> np.ndarray:
    
    """
    Returns the Weyl operator associated with index i in a d x d phase-space
    We order the Weyl operators as i -> (xi_x = i // d, xi_z = i % d).
    """
    
    if i >= d**2:
        raise ValueError("Index i out of range: i < d**2")

    xi_x = i // d
    xi_z = i % d
    return weyl(xi_x, xi_z, d)


# We can now easily obtain the superoperator in the Weyl basis. Notice that this gives information only within the defined computational subspace and not on the leakage subspace. Thus the following superoperator is complete knowledge of the states within the computational subspaces only if the leakage is zero. However, the following superoperator is the one needed to compute process fidelities as described in Eq. 8 of Wood-Gambetta Phys. Rev. A 97, 032306 (2018). The method to_super in the TomoEnv class is parallelized and we can specify the number of processes. 

# In[14]:


n_process = 4

start = time.time()
sup_op = env_syst.to_super(comp_states, weyl_by_index, n_process)
end = time.time()
print("Computational time = " + str(end - start) + ' s')


# Notice how the superoperator can be obtained for any system and its operations are completely system agnostic.
# 
# We now use the previous superoperator to compute the process fidelity and average gate fidelity with respect to a $\mathrm{CZ}$ gate. First of all, we need to obtain the single-qubit gate corrections. This can be done directly from the superoperator via the following function.

# In[15]:


def single_qubit_corrections(
    sup_op: np.ndarray,
    hs_basis: Callable[[int, int], np.ndarray]
) -> np.ndarray:
    sigma_m1 = tensor_prod([np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])])
    sigma_m2 = tensor_prod([np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])])
    sigma_m1_vec = trf.mat_to_vec(sigma_m1, hs_basis)
    sigma_m2_vec = trf.mat_to_vec(sigma_m2, hs_basis)
    evolved_sigma_m1_vec = sup_op.dot(sigma_m1_vec)
    evolved_sigma_m2_vec = sup_op.dot(sigma_m2_vec)
    evolved_sigma_m1 = trf.vec_to_mat(evolved_sigma_m1_vec, hs_basis)
    evolved_sigma_m2 = trf.vec_to_mat(evolved_sigma_m2_vec, hs_basis)
    phi10 = cmath.phase(hilbert_schmidt_prod(sigma_m1, evolved_sigma_m1))
    phi01 = cmath.phase(hilbert_schmidt_prod(sigma_m2, evolved_sigma_m2))
    p_phi10 = np.array([[1, 0], [0, np.exp(-1j*phi10)]])
    p_phi01 = np.array([[1, 0], [0, np.exp(-1j*phi01)]])
    return tensor_prod([p_phi10, p_phi01])


# We now compute the process fidelity. In order to do this we first need to obtain the superoperator representation of the $\mathrm{CZ}$ in the Weyl basis. This can be done via the function qutip.util.transformations.kraus_to_super

# In[16]:


cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
cz_super = trf.kraus_to_super(cz, weyl_by_index)


# We also obtain the superoperator representation of the single-qubit corrections and obtain the total applied superoperator

# In[17]:


sq_corr = single_qubit_corrections(sup_op, weyl_by_index)
sq_corr_sup = trf.kraus_to_super(sq_corr, weyl_by_index)
total_sup_op = sq_corr_sup.dot(sup_op)


# We can now compute the process fidelity

# In[18]:


f_pro = average_process_fidelity(cz_super, total_sup_op)
display(Latex(r'$F_{{pro}} = {:.5f}$'.format(f_pro)))


# Notice that this is by choice not done within the TomoEnv class, since the TomoEnv class should be agnostic of which gate we want to implement.
# 
# The average leakage and seepage is instead computed within the TomoEnv class. This is because we just need to specify a subspace to determine them, similarly to the determination of the superoperator. We use Eq. 2 and 3 in Wood-Gambetta Phys. Rev. A 97, 032306 (2018) to calculate the average leakage and seepage respectively.

# In[19]:


avg_leakage = env_syst.leakage(comp_states)
display(Latex(r'$L_1(\mathcal{{E}} ) = {:.7f}$'.format(avg_leakage)))


# In[20]:


avg_seepage = env_syst.seepage(comp_states)
display(Latex(r'$L_2(\mathcal{{E}} ) = {:.7f}$'.format(avg_seepage)))


# We can also check that with the selected accuracy of the qutip solver, for unitary dynamics (set loss tangents equal to zero) we reproduce the relation between leakage and seepage $d_1 L_1(\mathcal{E}) = d_2 L_2(\mathcal{E})$ ($d_1=4, d_2 = 11$)

# In[21]:


display(Latex(r'$d_1 L_1(\mathcal{{E}} ) = {:.7f}$'.format(d_comp*avg_leakage)))


# In[22]:


display(Latex(r'$d_2 L_2(\mathcal{{E}} ) = {:.7f}$'.format(d_leak*avg_seepage)))


# Finally, the average gate fidelity is

# In[23]:


f_gate = average_gate_fidelity(cz_super, total_sup_op, avg_leakage)
display(Latex(r'$F_{{gate}} = {:.5f}$'.format(f_gate)))

