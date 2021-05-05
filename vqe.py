from typing import Optional, Union, Callable, cast
import matplotlib.pyplot as plt
import numpy as np

from qiskit import IBMQ, BasicAer, Aer
from qiskit.providers.aer import StatevectorSimulator
from qiskit.utils import QuantumInstance

from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule, QMolecule
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper

from qiskit_nature.circuit.library import HartreeFock, UCC, UCCSD

from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import SLSQP


class Molecular_VQE():

	def __init__(self):
		# H2
		self.molecule_name = "Li .0 .0 .0; H .0 .0 "
		self.backend = QuantumInstance(backend=Aer.get_backend("statevector_simulator"))
		self.optimizer = SLSQP(maxiter=5)
		self.vqe = VQE(ansatz=None,
                       quantum_instance=self.backend,
                       optimizer=self.optimizer)
		#self.vqe_solver = VQEUCCFactory(self.backend)
		

	def get_qubit_op(self, dist, mapper='jw'):
		# Use PySCF, a classical computational chemistry software
		# package, to compute the one-body and two-body integrals in
		# electronic-orbital basis, necessary to form the Fermionic operator
		driver = PySCFDriver(atom=self.molecule_name + str(dist), unit=UnitsType.ANGSTROM, 
							 charge=0, spin=0, basis='sto3g')
		molecule = driver.run()
		es_problem = ElectronicStructureProblem(driver)

		if mapper == 'jw':
			qubit_converter = QubitConverter(mapper=JordanWignerMapper())
		elif mapper == 'parity':
			qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)

		#second_q_ops = es_problem.second_q_ops()
		#num_particles = es_problem.num_particles
		#molecule_data = es_problem.molecule_data
		#electronic_operator = second_q_ops[0]

		return es_problem, qubit_converter, molecule.nuclear_repulsion_energy


	def run(self):

		numpy_solver = NumPyMinimumEigensolver()

		distances = np.arange(0.3, 2.0, 0.05)
		exact_energies = []
		vqe_energies = []

		n = len(distances)
		i = 1

		for dist in distances:

			print("Distance %d/%d" % (i, n))
			i += 1
			
			es_problem, qubit_converter, nuclear_repulsion_energy = self.get_qubit_op(dist)
			second_q_ops = es_problem.second_q_ops()

			main_op = qubit_converter.convert(second_q_ops[0],
											  num_particles=es_problem.num_particles,
											  sector_locator=es_problem.symmetry_sector_locator)
			aux_ops = qubit_converter.convert_match(second_q_ops[1:])


			q_molecule_transformed = cast(QMolecule, es_problem.molecule_data_transformed)
			num_molecular_orbitals = q_molecule_transformed.num_molecular_orbitals
			num_particles = (q_molecule_transformed.num_alpha, q_molecule_transformed.num_beta)
			num_spin_orbitals = 2 * num_molecular_orbitals

			# Initial state is Hartree Fock state
			initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)

			# UCCSD ansatz for unitary update
			ansatz = UCCSD()
			ansatz.qubit_converter = qubit_converter
			ansatz.num_particles = num_particles
			ansatz.num_spin_orbitals = num_spin_orbitals
			ansatz.initial_state = initial_state

			self.vqe.ansatz = ansatz
			solver = self.vqe

			print("Computing minimum eigenvalue...")

			# Approximate minimum eigenvalue using VQE
			vqe_result = solver.compute_minimum_eigenvalue(main_op)
			print(np.real(vqe_result.eigenvalue) + nuclear_repulsion_energy)
			#exact_result = numpy_solver.compute_minimum_eigenvalue(main_op)

			vqe_energies.append(np.real(vqe_result.eigenvalue) + nuclear_repulsion_energy)
			#exact_energies.append(np.real(exact_result.eigenvalue) + nuclear_repulsion_energy)
			#print("Interatomic Distance:", np.round(dist, 2), "VQE Result:", vqe_result, "Exact Energy:", exact_energies[-1])

		return (distances, vqe_energies, exact_energies)

vqe = Molecular_VQE()
res = vqe.run()
plt.plot(res[0], res[1], c='r', label='VQE')
#plt.plot(res[0], res[2], c='b', label='Exact')
plt.rc('text', usetex=True)
plt.title(r"$LiH$: Minimum energy vs. interatomic distance")
plt.ylabel('Ground state energy (Hartree)')
plt.xlabel('Interatomic distance (A)')
plt.legend()
plt.show()

idx = res[1].index(min(res[1]))
dist, min_energy = res[0][res[1].index(min(res[1]))], min(res[1])
print("Distance: %f" % dist)
print("Energy: %f" % min_energy)






