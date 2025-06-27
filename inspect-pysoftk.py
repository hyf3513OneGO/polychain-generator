from rdkit import Chem
from rdkit.Chem import AllChem
from pysoftk.linear_polymer.linear_polymer import *
from pysoftk.format_printers.format_mol import *

def psmiles2ChainPoly(psmiles, n_repeat, save_path, relaxation_steps=350,force_filed="MMFF"):
    """
    Build a linear polymer from a given polymer SMILES and save it as an .sdf file (with bonding information).

    Parameters:
        psmiles (str): Polymer SMILES containing two [*] connection points.
        n_repeat (int): Number of repeating monomer units.
        save_path (str): Output path for the .mol2 file (should end with `.mol2`).
        relaxation_steps (int): Number of MMFF optimization steps.
    """
    a = Chem.MolFromSmiles(psmiles)
    AllChem.EmbedMolecule(a)

    chain = Lp(a, "*", n_copies=n_repeat, shift=2.5).linear_polymer(force_filed, relaxation_steps)
    # 假设 mol 是 Pybel.Molecule
    chain.OBMol.PerceiveBondOrders()  # 可选，确保键信息完整
    chain.OBMol.AddHydrogens()        # 可选，确保力场适配
    # chain.write("mol2", "chain.mol2", overwrite=True)
    chain.write("sdf", "chain.sdf", overwrite=True)

if __name__ == "__main__":
    psmiles2ChainPoly("[*]CC(CC([*])c1ccccc1)c1ccccc1", 5, "chain.mol",relaxation_steps=350)
