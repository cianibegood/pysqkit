import qutip as qtp 
import numpy as np 
from typing import List

list_Qobj = List[qtp.qobj.Qobj]
list_complex = List[complex] # could be complex in case we do RWA 

supported_solvers = ["mesolve"]

def integrate(
       tlist: np.ndarray,
       state_in: qtp.qobj.Qobj, #it can be given as ket or density matrix
       hamil0: qtp.qobj.Qobj, 
       drive: list_Qobj,
       pulse: list_complex, # we could add the option to give it analitically
       jump_op: list_Qobj,
       solver: str
) -> qtp.solver.Result:

    if len(drive) != len(pulse):
        raise ValueError("The number drive operators must be equal to the " \
            + "number of pulses!")

    if solver not in supported_solvers:
        raise ValueError("Unsupported solver. The solver must be in " + \
            + str(supported_solvers))

    #basic checks such as dimensionality will be automatically done by qutip

    h = [hamil0]
    for k in range(0, len(drive)):
        h.append([drive[k], pulse[k]])
    
    if solver == "mesolve":
        result = qtp.mesolve(h, state_in, tlist, jump_op)
    
    return result 
