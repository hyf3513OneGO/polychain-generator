from rdkit import Chem
from rdkit.Chem import AllChem
# from pysoftk.linear_polymer.linear_polymer import Lp
from utils.linear_polymer_utils import Lp
from pysoftk.format_printers.format_mol import *
import time
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

    for _ in range(10):
        if os.path.exists(save_path):
            break
        time.sleep(0.5)
    else:
        raise IOError(f"chain.write did not produce file: {save_path}")

if __name__ == "__main__":
    psmiles2ChainPoly("[*]CC(CC([*])c1ccccc1)c1ccccc1", 5, "chain.sdf",relaxation_steps=0)
