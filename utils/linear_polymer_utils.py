import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem.rdMolTransforms import *
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from pysoftk.tools.utils_func import *
from pysoftk.tools.utils_ob import *
from pysoftk.tools.utils_rdkit import *

from openbabel import openbabel as ob
from openbabel import openbabel
from openbabel import pybel as pb

def ff_ob_relaxation_rdkit(mol, FF="MMFF94", relax_iterations=100, ff_thr=1.0e-6):
    """
    基于 RDKit 的并行力场优化，接口与原函数一致（pybel.Molecule in/out）。
    内部使用 RDKit C++ 线程池，不占满 Python GIL。
    """
    # 1. SDF 序列化 -> RDKit Mol
    obconv = openbabel.OBConversion()
    obconv.SetOutFormat("sdf")
    sdf_block = obconv.WriteString(mol.OBMol)
    rdmol = Chem.MolFromMolBlock(sdf_block, sanitize=False, removeHs=False)
    Chem.SanitizeMol(rdmol)
    # 2. 坐标同步
    conf = rdmol.GetConformer()
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        ob_atom = mol.OBMol.GetAtom(idx+1)
        conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(
            ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()))

    # 3. 优化
    n_threads = os.cpu_count() or 1
    energies = []
    if FF.upper().startswith("MMFF"):
        props = AllChem.MMFFGetMoleculeProperties(rdmol, mmffVariant=FF)
        energies = AllChem.MMFFOptimizeMoleculeConfs(
            rdmol,
            numThreads=n_threads,
            maxIters=relax_iterations,
            nonBondedThresh=ff_thr,
            mmffVariant=FF
        )
    elif FF.upper() == "UFF":
        energies = AllChem.UFFOptimizeMoleculeConfs(
            rdmol,
            numThreads=n_threads,
            maxIters=relax_iterations
        )
    else:
        raise ValueError(f"Unsupported FF: {FF}")

    # 4. 坐标回写
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        ob_atom = mol.OBMol.GetAtom(idx+1)
        ob_atom.SetVector(pos.x, pos.y, pos.z)

    return mol
class Lp:
    """
    A class for constructing linear polymers from individual molecular units (monomers)
    using RDKit functionalities.

    This class facilitates the creation of polymeric structures by taking a monomer
    RDKit molecule, replicating it a specified number of times, and then combining
    these copies into a linear chain. It handles the spatial arrangement and bonding
    of the monomer units, including the removal of placeholder atoms and subsequent
    geometry optimization using force fields.

    Examples
    ---------
    >>> from rdkit import Chem
    >>> from pysoftk.linear_polymers.linear_polymer import Lp
    >>> # Example: Create a simple monomer with a placeholder atom (e.g., [Pt])
    >>> monomer_smiles = "C([Pt])C"
    >>> monomer = Chem.MolFromSmiles(monomer_smiles)
    >>> # Initialize Lp with the monomer, placeholder atom, number of copies, and shift
    >>> linear_poly_builder = Lp(mol=monomer, atom="Pt", n_copies=3, shift=1.5)
    >>> # Generate the linear polymer with MMFF force field
    >>> polymer_mol = linear_poly_builder.linear_polymer(force_field="MMFF")
    >>> print(f"Polymer atoms: {polymer_mol.GetNumAtoms()}")
    >>> print(f"Polymer bonds: {polymer_mol.GetNumBonds()}")

    Note
    -----
    The RDKit and OpenBabel Python packages must be installed for this class to function
    correctly. Ensure that any placeholder atoms used in the monomer SMILES
    (e.g., "[Pt]") are consistent with the `atom` parameter.
    """

    __slots__ = ['mol', 'atom', 'n_copies', 'shift']

    def __init__(self, mol, atom, n_copies, shift):
        """
        Initializes the Lp class with the monomer molecule, placeholder atom,
        number of copies, and spatial shift.

        Parameters
        -----------
        mol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object representing the monomer unit. This molecule
            should contain a designated placeholder atom that will be used for
            connecting the monomer copies.

        atom : str
            The atomic symbol of the placeholder atom in the `mol` that will be
            removed and replaced with a bond to form the polymer chain (e.g., "Pt", "Au").

        n_copies : int
            The total number of monomer units to be linked together to form the
            linear polymer.

        shift : float
            The initial X-axis translation distance (in Ångstroms) applied to
            each subsequent monomer copy before bond formation. This value
            influences the initial spacing between monomer units. If `None`,
            a default value of 1.25 Å is used.
        """

        self.mol = mol
        self.atom = atom
        self.n_copies = n_copies
        self.shift = float(1.25) if shift is None else float(shift)

    def max_dist_mol(self):
        """
        Calculates and returns the maximum interatomic distance within the
        initial RDKit monomer molecule (`self.mol`).

        This distance is crucial for determining the appropriate spacing
        between repeated monomer units to avoid steric clashes during the
        initial polymer assembly.

        Return
        -------
        np.float
            The largest distance (in Ångstroms) found between any two atoms
            in the monomer's conformational representation.
        """
        
        mol = self.mol
        bm = molDG.GetMoleculeBoundsMatrix(mol)
        return np.amax(bm)

    def x_shift(self):
        """
        Computes the refined X-axis translation value for positioning
        subsequent monomer units during polymer construction.

        This method adjusts the initial `shift` value based on the maximum
        distance between atoms in the monomer, aiming to create a more
        realistic and physically sound initial arrangement of the polymer chain.

        Return
        -------
        shift_final : float
            The calculated X-axis translation value (in Ångstroms) that will be
            applied to each repeating unit to properly space them in the linear polymer.
        """

        shift = self.shift
        shift_final = float(self.max_dist_mol()) - float(shift)

        return shift_final

    def copy_mol(self):
        """
        Generates a list of identical RDKit molecular objects, each representing
        a copy of the initial monomer.

        Before replication, the conformer of the original molecule is canonicalized
        to ensure consistent orientation for all copies.

        Parameters
        -----------
        mol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object (self.mol) that will be replicated.

        Return
        --------
        fragments : list[rdkit.Chem.rdchem.Mol]
            A list containing `n_copies` identical RDKit Mol objects, each
            ready to be incorporated into the polymer chain.
        """

        mol = self.mol
        CanonicalizeConformer(mol.GetConformer())

        n_copies = self.n_copies

        fragments = [mol for _ in range(int(n_copies))]

        return fragments

    def polimerisation(self, fragments):
        """
        Recursively combines a list of RDKit monomer fragments into a single
        linear polymer RDKit molecule.

        Each subsequent fragment is combined with the growing polymer chain,
        with a calculated X-axis offset to maintain linearity and avoid overlap.
        The atoms in the resulting molecule are then renumbered for canonical ordering.

        Parameters
        -----------
        fragments : list[rdkit.Chem.rdchem.Mol]
            A list of RDKit Mol objects, each representing a monomer unit
            to be joined to form the polymer.

        Return
        --------
        outmol : rdkit.Chem.rdchem.Mol
            A single RDKit Mol object representing the combined linear polymer
            with all monomer units spatially arranged and canonically ordered.
        """

        x_offset = self.x_shift()

        outmol = fragments[0]
        for idx, values in enumerate(fragments[1:]):
            outmol = Chem.CombineMols(outmol, values,
                                      offset=Point3D(x_offset * (idx + 1), 0.0, 0.0))

        order = Chem.CanonicalRankAtoms(outmol, includeChirality=True)
        mol_ordered = Chem.RenumberAtoms(outmol, list(order))

        return outmol

    def bond_conn(self, outmol):
        """
        Identifies and processes the bonding information related to the placeholder
        atom within the combined super-monomer structure (`outmol`).

        This method extracts information about the atoms that are bonded to the
        placeholder atom, which is crucial for forming new bonds between monomers
        and subsequently removing the placeholder.

        Parameters
        -----------
        outmol : rdkit.Chem.rdchem.Mol
            The RDKit Mol object representing the partially formed polymer
            (combined super-monomer), containing the placeholder atoms.

        Return
        --------
        Tuple : (list[tuple], list[int])
            A tuple containing two lists:
            - The first list (`all_conn`) contains tuples of atom indices
              that represent the connections to be made between the monomers
              after the placeholder atoms are removed.
            - The second list (`erase_br`) contains the indices of the
              placeholder atoms that need to be removed from the molecule.
        """
        
        atom = self.atom

        bonds = atom_neigh(outmol, str(atom))
        conn_bonds = [b for a, b in bonds][1:-1]

        erase_br = [a for a, b in bonds]
        all_conn = list(zip(conn_bonds[::2], conn_bonds[1::2]))

        return all_conn, erase_br

    def proto_polymer(self):
        """
        Constructs the initial "proto-polymer" by combining monomer units,
        forming new bonds, and removing the placeholder atoms.

        This method orchestrates the steps of copying monomers, spatially
        arranging them, identifying and creating new bonds between them,
        and finally removing the temporary placeholder atoms used for connection.
        Hydrogen atoms are also added to the resulting molecule.

        Returns
        --------
        newMol_H : rdkit.Chem.rdchem.Mol
            A new RDKit Mol object representing the raw linear polymer structure,
            with placeholder atoms removed and new bonds formed, and explicit
            hydrogen atoms added with their coordinates.
        """
        
        atom = self.atom
        mol = self.mol
        n_copies = self.n_copies

        fragments = self.copy_mol()
        outmol = self.polimerisation(fragments)
        all_conn, erase_br = self.bond_conn(outmol)

        rwmol = Chem.RWMol(outmol)
        for ini, fin in all_conn:
            rwmol.AddBond(ini, fin, Chem.BondType.SINGLE)

        for i in sorted(erase_br[1:-1], key=None, reverse=True):
            rwmol.RemoveAtom(i)

        mol3 = rwmol.GetMol()
        Chem.SanitizeMol(mol3)

        mol4 = Chem.AddHs(mol3, addCoords=True)

        return mol4

    def linear_polymer(self, force_field="MMFF", relax_iterations=350, rot_steps=125, no_att=True):
        """
        Generates and optimizes the 3D structure of a linear polymer from the
        provided monomer unit.

        This is the main method for creating the final polymer. It first constructs
        a "proto-polymer" by combining monomers and forming initial bonds. Then,
        it can optionally remove any remaining placeholder atoms. Finally, it
        applies a specified force field to relax the geometry and performs
        rotor optimization to refine the polymer's conformation.

        Parameters
        -----------
        force_field : str, optional
            The name of the force field to use for geometry optimization.
            Accepted values are "MMFF", "UFF", or "MMFF94". "MMFF" will
            automatically default to "MMFF94". Defaults to "MMFF".
        relax_iterations : int, optional
            The maximum number of iterations to use during the force field
            relaxation step. A higher number generally leads to better
            minimization but takes longer. Defaults to 350.
        rot_steps : int, optional
            The number of steps for rotor optimization, which involves
            optimizing dihedral angles to find low-energy conformers.
            Defaults to 125.
        no_att : bool, optional
            If `True`, any remaining placeholder atoms (as defined by `self.atom`)
            will be removed from the polymer structure before force field
            optimization. If `False`, placeholder atoms will be retained.
            Defaults to `True`.

        Returns
        --------
        newMol_H : openbabel.pybel.Molecule
            An OpenBabel Molecule object representing the optimized 3D structure
            of the linear polymer.

        Raises
        ------
        ValueError
            If an invalid `force_field` is provided or if `relax_iterations`
            or `rot_steps` are not integers.
        """

        mol = self.proto_polymer()
        atom = self.atom

        if no_att:
            mol1 = remove_plcholder(mol, atom)
        else:
            mol1 = mol  # Use the original mol if mol1 is not provided

        # Using PDB object to preserve bond information
        last_rdkit = Chem.MolToPDBBlock(mol1)
        mol_new = pb.readstring('pdb', last_rdkit)

        # Validate force field:
        valid_force_fields = ("MMFF", "UFF", "MMFF94")
        if force_field not in valid_force_fields:
            raise ValueError(f"Invalid force field: {force_field}. Valid options are: {valid_force_fields}")

        # Automatically change ff if necessary:
        if force_field == "MMFF":
            force_field = "MMFF94"  # Change to default MMFF94 for MMFF

        # Validate and convert iterations and steps to integers:
        try:
            relax_iterations = int(relax_iterations)
            rot_steps = int(rot_steps)
        except ValueError:
            raise ValueError("relax_iterations and rot_steps must be integers.")

        # Relaxation and optimization:
        if relax_iterations >0:
            last_mol = ff_ob_relaxation_rdkit(mol_new, force_field, relax_iterations)
            rot_mol = rotor_opt(last_mol, force_field, rot_steps)

            return rot_mol
        else:
            # If no relaxation is needed, return the initial molecule
            return mol_new