import qutip as qtp 
import numpy as np 
from typing import List

supported_solvers = ["mesolve"]

def integrate(
       tlist: np.ndarray,
       state_in: qtp.qobj.Qobj, #it can be given as ket or density matrix
       hamil0: qtp.qobj.Qobj, 
       drive: List[qtp.qobj.Qobj],
       pulse: List[np.ndarray], # we could add the option to give it analitically
       jump_op: List[qtp.qobj.Qobj],
       solver: str
) -> qtp.solver.Result:

    if len(drive) != len(pulse):
        raise ValueError("The number of drive operators must be equal to the " + \
            "number of pulses!")

    if solver not in supported_solvers:
        raise ValueError("Unsupported solver. The solver must be in " + \
            str(supported_solvers))

    #basic checks such as dimensionality will be automatically done by qutip

    h = [hamil0]
    for k in range(0, len(drive)):
        h.append([drive[k], pulse[k]])
    
    if solver == "mesolve":
        result = qtp.mesolve(h, state_in, tlist, jump_op)
    
    return result 

def cptp_map(
        tlist: np.ndarray,
        hamil0: qtp.qobj.Qobj, 
        drive: List[qtp.qobj.Qobj],
        pulse_func: list, # it is a list of functions
        jump_op: List[qtp.qobj.Qobj], 
        args: dict = None
) -> qtp.qobj.Qobj:
    #TO DO: allow to return the map at user-defined points, 
    #not necessarily at tlist

    if len(drive) != len(pulse_func):
        raise ValueError("The number of drive operators must be equal to the " + \
            "number of pulses!")
    
    hamil = [hamil0]
    for k in range(0, len(drive)):
        hamil.append([drive[k], pulse_func[k]])
    hamil_asqobjevo = qtp.QobjEvo(hamil, args) 

    liouv = qtp.liouvillian(hamil_asqobjevo, jump_op)
    liouv_in = liouv(0)
    identity_map = qtp.Qobj(
        inpt=qtp.qeye(liouv_in.shape[0]),
        dims=liouv_in.dims,
        shape=liouv_in.shape,
        type='super',
        isherm='True'
    )

    eps_map = identity_map
    eps_map_kraus = [qtp.to_kraus(eps_map)]
    for k in range(0, len(tlist) - 1):
        delta_t = tlist[k + 1] - tlist[k]
        eps_map = (identity_map + delta_t*liouv(tlist[k]))*eps_map
        eps_map_kraus.append(qtp.to_kraus(eps_map))
    return eps_map_kraus 


    
    
    
