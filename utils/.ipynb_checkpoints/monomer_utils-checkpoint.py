from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
import os

def remove_star_atoms(monomer_smiles):
	smi = monomer_smiles
	mol = Chem.MolFromSmiles(smi)

	assert 'H' not in [atom.GetSymbol() for atom in mol.GetAtoms()]

	key_point_list = []
	for atom in mol.GetAtoms():
		# 从邻居节点中去找*，然后把自身放到*的位置上  br->* *->br psmiles->smiles
		atom_neighbors = atom.GetNeighbors()
		for atom_neighbor in atom_neighbors:
			if atom_neighbor.GetSymbol() == '*':
				key_point_list.append(atom.GetIdx())
				key_point_list.append(atom_neighbor.GetIdx())

	assert len(key_point_list) == 4, 'Unvalid PSMILES'

	star_0 = key_point_list[1]
	star_1 = key_point_list[3]
	neighbor_0 = key_point_list[0]
	neighbor_1 = key_point_list[2]

	atom = mol.GetAtomWithIdx(star_0)
	atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_1).GetAtomicNum())

	atom = mol.GetAtomWithIdx(star_1)
	atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_0).GetAtomicNum())

	processed_smi = ""
	replacement = [str(mol.GetAtomWithIdx(neighbor_1).GetSymbol()), str(mol.GetAtomWithIdx(neighbor_0).GetSymbol())]
	count = 0
	i = 0
	while i != len(smi):
		if smi[i:i + 3] == "[*]":
			processed_smi += replacement[count]
			count += 1
			i += 3
		else:
			processed_smi += smi[i]
			i += 1

	pre_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
	processed_mol = Chem.MolFromSmiles(processed_smi)
	post_atoms = [atom.GetSymbol() for atom in processed_mol.GetAtoms()]

	assert pre_atoms == post_atoms, 'Unmatch Order'

	return key_point_list, processed_mol,processed_smi

    

def extract_valid_submol(mol, atom_indices):
    """Extract valid submol given atom indices, including bonds."""
    atom_indices = sorted(set(idx for idx in atom_indices if idx < mol.GetNumAtoms()))
    if not atom_indices:
        return None
    emol = Chem.EditableMol(Chem.Mol())
    idx_map = {}

    for i, idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_idx = emol.AddAtom(new_atom)
        idx_map[idx] = new_idx

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in idx_map and a2 in idx_map:
            emol.AddBond(idx_map[a1], idx_map[a2], bond.GetBondType())

    submol = emol.GetMol()
    AllChem.Compute2DCoords(submol)
    return submol

def extract_repeat_units_from_sdf(polymer_sdf_path, monomer_psmiles, output_dir, max_units=None):
    """Extract non-overlapping monomers from a polymer SDF, matching a polymer SMILES template."""
    os.makedirs(output_dir, exist_ok=True)

    # Load polymer and prepare template
    polymer = Chem.MolFromMolFile(polymer_sdf_path, removeHs=False)
    if polymer is None:
        raise ValueError("Cannot load polymer SDF file.")

    monomer_query = Chem.MolFromSmiles(monomer_psmiles)
    monomer_core = remove_star_atoms(monomer_query)

    # Find all matches, and filter non-overlapping
    all_matches = polymer.GetSubstructMatches(monomer_core)
    used_atoms = set()
    non_overlap_matches = []
    for match in all_matches:
        if not any(idx in used_atoms for idx in match):
            non_overlap_matches.append(match)
            used_atoms.update(match)
        if max_units and len(non_overlap_matches) >= max_units:
            break

    # Extract and write
    sdf_paths = []
    for i, match in enumerate(non_overlap_matches):
        atom_set = set(match)
        for idx in match:
            atom = polymer.GetAtomWithIdx(idx)
            atom_set.update(n.GetIdx() for n in atom.GetNeighbors())

        submol = extract_valid_submol(polymer, atom_set)
        if submol:
            out_path = os.path.join(output_dir, f"monomer_{i+1:02d}.sdf")
            writer = SDWriter(out_path)
            writer.write(submol)
            writer.close()
            sdf_paths.append(out_path)

    return sdf_paths

from rdkit import Chem
from rdkit.Chem import Draw
import os

from rdkit import Chem
from rdkit.Chem import Draw
import os

def visualize_monomer_sdf_grid(sdf_paths, mols_per_row=5, size=(300, 300), use_svg=False):
    """
    可视化一组 monomer 的 SDF 文件为网格图像。

    参数：
    - sdf_paths: list of str, 每个 .sdf 文件的路径
    - mols_per_row: int, 每行显示的 monomer 数量
    - size: tuple(int, int), 每个单体图片大小
    - use_svg: bool, 是否使用 SVG 图像格式（推荐在 Jupyter 中使用）

    返回：
    - PIL.Image.Image 或 SVG 字符串（取决于 use_svg）
    """
    mols = [Chem.MolFromMolFile(p, removeHs=False) for p in sdf_paths]
    mols = [m for m in mols if m is not None]
    legends = [f"Monomer {i+1}" for i in range(len(mols))]
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size, legends=legends, useSVG=use_svg)
    return img

from rdkit import Chem
import numpy as np
import json

def get_connection_atoms(monomer_smiles):
    """
    提取 SMILES 中的两个连接点（[*]）在原始分子中的索引。
    """
    mol = Chem.MolFromSmiles(monomer_smiles)
    conn_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(conn_indices) != 2:
        raise ValueError("SMILES 中连接点 [*] 数量不是 2")
    return conn_indices

def compute_frame_from_connection_atoms(mol, conn_atom_idx1, conn_atom_idx2, center_atom_idx=None):
    """
    使用两个连接原子构造局部刚体坐标系 Frame（R, t）
    R: 旋转矩阵，t: 平移向量
    """
    conf = mol.GetConformer()
    p1 = np.array(conf.GetAtomPosition(conn_atom_idx1))
    p2 = np.array(conf.GetAtomPosition(conn_atom_idx2))
    v1 = p2 - p1

    # x 轴
    x_axis = v1 / np.linalg.norm(v1)
    # 任意参考向量构造 y
    ref = np.array([0, 0, 1.0]) if abs(x_axis[2]) < 0.9 else np.array([1.0, 0, 0])
    y_temp = np.cross(x_axis, ref)
    y_axis = y_temp / np.linalg.norm(y_temp)
    # z 轴
    z_axis = np.cross(x_axis, y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    t = np.array(conf.GetAtomPosition(center_atom_idx)) if center_atom_idx is not None else p1

    return R, t

def extract_SE3_frames_from_monomers(mol_list, psmiles, center_atom_idx=None):
    """
    批量提取每个 monomer 的刚体 frame (R, t)，基于 pSMILES 中连接点定义。
    返回 List[dict] 格式的刚体参数。
    """
    conn1, conn2 = get_connection_atoms(psmiles)
    frames = []
    for i, mol in enumerate(mol_list):
        try:
            R, t = compute_frame_from_connection_atoms(mol, conn1, conn2, center_atom_idx)
            frames.append({
                "R": R.tolist(),
                "t": t.tolist()
            })
        except Exception as e:
            print(f"[Warning] Failed to extract frame for monomer {i+1}: {str(e)}")
            frames.append(None)
    return frames

def save_SE3_frames_to_json(frames, output_path):
    with open(output_path, "w") as f:
        json.dump(frames, f, indent=2)




from rdkit import Chem
import numpy as np

def apply_SE3_to_mol(mol, R, t):
    """
    将刚体变换 (R, t) 应用于 mol 的 conformer 原子坐标
    """
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = R @ pos + t
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*new_pos))
    return mol
def check_valence_and_add_bond(mol, a1_idx, a2_idx):
    """
    检查两个原子是否可以添加连接键，避免原子的价键超过最大限制。
    """
    # 获取原子的当前价键数
    a1 = mol.GetAtomWithIdx(a1_idx)
    a2 = mol.GetAtomWithIdx(a2_idx)

    if a1.GetNumImplicitHs() + a1.GetDegree() >= a1.GetMaxNumNeighbors():
        raise ValueError(f"Atom {a1_idx} exceeds valence!")
    if a2.GetNumImplicitHs() + a2.GetDegree() >= a2.GetMaxNumNeighbors():
        raise ValueError(f"Atom {a2_idx} exceeds valence!")

    mol.AddBond(a1_idx, a2_idx, Chem.BondType.SINGLE)
def reconstruct_polymer_from_monomers(mol_list, frames, bond_map):
    """
    使用 frames 中的 (R, t) 刚体变换将 monomer 拼接为一个聚合物分子。
    确保每个连接点只添加一次（共享原子）。
    """
    combined = Chem.RWMol()
    offset = 0
    for i, (mol, frame) in enumerate(zip(mol_list, frames)):
        if frame is None:
            continue
        R = np.array(frame["R"])
        t = np.array(frame["t"])
        mol = Chem.Mol(mol)
        mol = apply_SE3_to_mol(mol, R, t)
        amap = {}  # 原子索引映射表
        for atom in mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            idx = combined.AddAtom(new_atom)
            amap[atom.GetIdx()] = idx

        # 添加键
        for bond in mol.GetBonds():
            a1 = amap[bond.GetBeginAtomIdx()]
            a2 = amap[bond.GetEndAtomIdx()]
            check_valence_and_add_bond(combined, a1, a2)

        # 连接相邻 monomer，添加连接键（避免重复添加连接点）
        if i > 0:
            bond_start, bond_end = bond_map[i - 1]
            # bond_map 中的连接点是共享原子（不重复添加）
            a1 = amap[bond_start]
            a2 = amap[bond_end]
            # 如果已经添加过这个连接键，就跳过
            if combined.GetBondBetweenAtoms(a1, a2) is None:
                check_valence_and_add_bond(combined, a1, a2)

        offset += mol.GetNumAtoms()

    combined.UpdatePropertyCache()
    Chem.SanitizeMol(combined)
    return combined

def get_connection_atoms_from_smiles(psmiles):
    """
    从 monomer SMILES 提取连接点 [*] 的原子索引（连接原子）。
    """
    mol = Chem.MolFromSmiles(psmiles)
    conn_indices = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            conn_indices.append(atom.GetIdx())
    
    if len(conn_indices) != 2:
        raise ValueError(f"SMILES 中连接点 [*] 数量不是 2: {psmiles}")
    
    return conn_indices

def generate_bond_map_from_smiles(psmiles_list):
    """
    根据 SMILES 列表生成连接键的 bond_map，
    bond_map 由每两个相邻 monomer 的连接点（[*]）构成。
    """
    bond_map = []
    for i in range(1, len(psmiles_list)):
        prev_conn_atoms = get_connection_atoms_from_smiles(psmiles_list[i-1])  # 上一个 monomer 的连接原子
        curr_conn_atoms = get_connection_atoms_from_smiles(psmiles_list[i])    # 当前 monomer 的连接原子

        # 使用前一个 monomer 的第二个连接原子（即末端）连接当前 monomer 的第一个连接原子
        bond_map.append((prev_conn_atoms[1], curr_conn_atoms[0]))
    
    return bond_map

