from rdkit import Chem
from rdkit.Chem import AllChem

def psmiles2ChainPoly(psmiles, n_repeat, save_path, relaxation_steps=350, force_field="MMFF"):
    """
    从含 dummy atom（[*]）的 psmiles 构建线性聚合物。
    自动识别连接原子，无需手动指定。
    
    参数:
        psmiles (str): 包含两个 [*] 连接位点的 monomer SMILES
        n_repeat (int): 重复次数
        save_path (str): 输出 SDF 路径
        relaxation_steps (int): 构象优化迭代步数
        force_field (str): "MMFF" 或 "UFF"
    """
    assert psmiles.count("*") == 2, "psmiles 中必须包含两个 dummy atom（[*]）"

    # 1. 从 psmiles 解析 monomer，识别连接位点原子
    mol_dummy = Chem.MolFromSmiles(psmiles)
    conn_atom_ids = []
    dummy_ids = []

    for atom in mol_dummy.GetAtoms():
        if atom.GetSymbol() == "*":
            dummy_ids.append(atom.GetIdx())
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1, "每个 dummy atom 应只连接一个原子"
            conn_atom_ids.append(neighbors[0].GetIdx())

    assert len(conn_atom_ids) == 2, "应识别到两个连接原子"

    # 2. 构造 monomer（删除 dummy atom，替换为 H）
    editable = Chem.EditableMol(mol_dummy)
    for idx in sorted(dummy_ids, reverse=True):
        editable.RemoveAtom(idx)
    monomer = editable.GetMol()
    monomer = Chem.AddHs(monomer)
    AllChem.EmbedMolecule(monomer)

    monomer_atom_count = monomer.GetNumAtoms()
    conn_atom1, conn_atom2 = conn_atom_ids  # 在原始 mol 中的原子索引
    if dummy_ids[0] < dummy_ids[1]:  # 如果删掉前面那个 dummy，后面索引会变
        conn_atom2 -= 1

    # 3. 构建聚合物链
    polymer = Chem.RWMol(monomer)
    monomer_ranges = [(0, monomer_atom_count)]

    for i in range(1, n_repeat):
        mol_cp = Chem.Mol(monomer)
        conf_cp = mol_cp.GetConformer()
        offset = polymer.GetNumAtoms()

        # 平移复制的 monomer（避免坐标重叠）
        for j in range(monomer_atom_count):
            pos = conf_cp.GetAtomPosition(j)
            conf_cp.SetAtomPosition(j, (pos.x + i * 3.0, pos.y, pos.z))

        combined = Chem.CombineMols(polymer, mol_cp)
        polymer = Chem.RWMol(combined)
        monomer_ranges.append((offset, offset + monomer_atom_count))

        # 添加连接键：尾→头
        a1 = monomer_ranges[i-1][0] + conn_atom2  # 上一个 monomer 的尾部连接位点
        a2 = monomer_ranges[i][0] + conn_atom1    # 当前 monomer 的头部连接位点
        polymer.AddBond(a1, a2, Chem.BondType.SINGLE)

    final = polymer.GetMol()

    # 4. 构象优化
    if relaxation_steps > 0:
        if force_field.upper() == "MMFF":
            AllChem.MMFFOptimizeMolecule(final, maxIters=relaxation_steps)
        elif force_field.upper() == "UFF":
            AllChem.UFFOptimizeMolecule(final, maxIters=relaxation_steps)
        else:
            raise ValueError("只支持 force_field='MMFF' 或 'UFF'")
    Chem.MolToMolFile(final, save_path)
    print(f"构建完成，已保存至：{save_path}")

if __name__ == "__main__":
    psmiles2ChainPoly(
        "[*]CCC[*]",
        n_repeat=5,
        save_path="results/chain_manual.sdf",
        relaxation_steps=350
    )