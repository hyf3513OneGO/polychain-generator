from rdkit import Chem
from rdkit.Chem import AllChem
from pysoftk.linear_polymer.linear_polymer import Lp
from pysoftk.format_printers.format_mol import *

def psmiles2ChainPoly(psmiles, n_repeat, save_path, relaxation_steps=350, force_field="MMFF"):
    # 拆成去掉 dummy atom 的构象生成
    monomer = Chem.MolFromSmiles(psmiles)

    # 构象嵌入
    AllChem.EmbedMolecule(monomer)

    # Construct polymer chain
    chain = Lp(monomer, "Pt", n_copies=n_repeat, shift=2.5).linear_polymer(force_field, relaxation_steps)

    chain.OBMol.PerceiveBondOrders()
    chain.OBMol.AddHydrogens()
    chain.write("sdf", save_path, overwrite=True)



if __name__ == "__main__":
    psmiles2ChainPoly("[*]CC(CC([*])c1ccccc1)c1ccccc1", 5, "chain.sdf",relaxation_steps=0)
