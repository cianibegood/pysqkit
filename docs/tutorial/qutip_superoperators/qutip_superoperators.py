#%%
import numpy as np 
import qutip as qtp 

#%%
""" Checking qutip superoperators some test operator op for
one qubit. The convention is that the operator op acts on 
another operator a as op*a*op.dag(). Also the convention is 
to order the operators by columns. This is because qutip defines 
the sigma_plus operators in the "quantum optics way""""

op = qtp.qobj.Qobj(np.array([[0, 2j], [3 -4j, 0]]))
# op = qtp.qobj.Qobj(np.array([[0, 10], [2, 0]]))
op_super = qtp.to_super(op)
print(op_super) 

#%%
""" Again notice that """
print("Qutip sigma_plus: \n" + str(qtp.sigmap()))
print("Qutip sigma_minus: \n" + str(qtp.sigmam()))

#%% 
""" The order of the basis according to the quantum
information convention is |0><0|, |1><0|,
|0><1|, |1><1|. In fact """

print(op*(qtp.qeye(2) + qtp.sigmaz())/2*op.dag())
print(op*qtp.sigmam()*op.dag())
print(op*qtp.sigmap()*op.dag())
print(op*(qtp.qeye(2) - qtp.sigmaz())/2*op.dag())

""" N.B. qtp.to_chi returns what we would call the Pauli
transfer matrix associated with the operator...looking at
the source code it is implemented only for qubits which is a big 
limitation. """





# %%
