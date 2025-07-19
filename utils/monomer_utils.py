from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
import os
import numpy as np
from rdkit import Chem
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import rdchem
import json
import os
def remove_star_atoms(monomer_smiles):
    mol = Chem.MolFromSmiles(monomer_smiles)
    assert 'H' not in [atom.GetSymbol() for atom in mol.GetAtoms()]

    key_point_list = []
    for atom in mol.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == '*':
                key_point_list.append(atom.GetIdx())
                key_point_list.append(neighbor.GetIdx())

    assert len(key_point_list) == 4, 'Invalid PSMILES: should have two [*] atoms.'

    star_0, star_1 = key_point_list[1], key_point_list[3]
    neighbor_0, neighbor_1 = key_point_list[0], key_point_list[2]

    # 修改原子类型以便还原真实连接
    mol.GetAtomWithIdx(star_0).SetAtomicNum(mol.GetAtomWithIdx(neighbor_1).GetAtomicNum())
    mol.GetAtomWithIdx(star_1).SetAtomicNum(mol.GetAtomWithIdx(neighbor_0).GetAtomicNum())

    # 用真实原子替换 SMILES 中的 [*]
    replacements = [mol.GetAtomWithIdx(neighbor_1).GetSymbol(), mol.GetAtomWithIdx(neighbor_0).GetSymbol()]
    processed_smi = ""
    count = 0
    i = 0
    while i < len(monomer_smiles):
        if monomer_smiles[i:i + 3] == "[*]":
            processed_smi += replacements[count]
            count += 1
            i += 3
        else:
            processed_smi += monomer_smiles[i]
            i += 1

    pre_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    processed_mol = Chem.MolFromSmiles(processed_smi)
    post_atoms = [atom.GetSymbol() for atom in processed_mol.GetAtoms()]

    assert pre_atoms == post_atoms, 'Atom order mismatch after replacement.'

    return key_point_list, processed_mol, processed_smi

def remove_star_atoms_withH(monomer_smiles):
    '''
    keypoint_list [n1,s1,n2,s2]
    '''
    mol = Chem.MolFromSmiles(monomer_smiles)
    mol = Chem.AddHs(mol)
    # assert 'H' not in [atom.GetSymbol() for atom in mol.GetAtoms()]

    key_point_list = []
    for atom in mol.GetAtoms():
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == '*' and atom.GetSymbol()!="H":
                key_point_list.append(atom.GetIdx())
                key_point_list.append(neighbor.GetIdx())

    assert len(key_point_list) == 4, 'Invalid PSMILES: should have two [*] atoms.'

    star_0, star_1 = key_point_list[1], key_point_list[3]
    neighbor_0, neighbor_1 = key_point_list[0], key_point_list[2]

    # 修改原子类型以便还原真实连接
    mol.GetAtomWithIdx(star_0).SetAtomicNum(mol.GetAtomWithIdx(neighbor_1).GetAtomicNum())
    mol.GetAtomWithIdx(star_1).SetAtomicNum(mol.GetAtomWithIdx(neighbor_0).GetAtomicNum())

    # 用真实原子替换 SMILES 中的 [*]
    replacements = [mol.GetAtomWithIdx(neighbor_1).GetSymbol(), mol.GetAtomWithIdx(neighbor_0).GetSymbol()]
    processed_smi = ""
    count = 0
    i = 0
    while i < len(monomer_smiles):
        if monomer_smiles[i:i + 3] == "[*]":
            processed_smi += replacements[count]
            count += 1
            i += 3
        else:
            processed_smi += monomer_smiles[i]
            i += 1
    mol_noH = Chem.RemoveHs(mol)
    pre_atoms = [atom.GetSymbol() for atom in mol_noH.GetAtoms()]
    processed_mol = Chem.MolFromSmiles(processed_smi)
    post_atoms = [atom.GetSymbol() for atom in processed_mol.GetAtoms()]
    assert pre_atoms == post_atoms, 'Atom order mismatch after replacement.'
    processed_mol = Chem.AddHs(processed_mol)
    return key_point_list, processed_mol, processed_smi
def extract_monomers(sdf_file, smiles):
    mol_H = Chem.MolFromMolFile(sdf_file, removeHs=False)
    mol_noH = Chem.RemoveHs(mol_H, updateExplicitCount=True)
    map_noH_to_H = backbone2full(mol_H, mol_noH)

    keypoints, processed_mol, processed_smi = remove_star_atoms(smiles)
    n1, star1, n2, star2 = keypoints  # n1-*1, n2-*2

    monomer_H = Chem.MolFromSmiles(processed_smi)
    monomer_noH = Chem.RemoveHs(monomer_H, updateExplicitCount=True)
    n_monomer_atoms = monomer_noH.GetNumAtoms()

    n_poly_atoms = mol_noH.GetNumAtoms()
    n_repeat = n_poly_atoms // (n_monomer_atoms - 2)
    total_index = list(range(n_poly_atoms))

    # Step 1: 初始原子编号分配
    filled_index = []
    cursor = 0
    for _ in range(n_repeat):
        idx_map = [-1] * n_monomer_atoms
        for j in range(n_monomer_atoms):
            if j != star1 and j != star2:
                idx_map[j] = total_index[cursor]
                cursor += 1
        filled_index.append(idx_map)

    # Step 2: 设置连接原子编号
    for i in range(n_repeat):
        if i > 0:
            filled_index[i][star1] = filled_index[i - 1][n2]
        if i < n_repeat - 1:
            filled_index[i][star2] = filled_index[i + 1][n1]

    # Step 3: 去除边界 monomer 中未补全原子
    filled_index_clean = [
        [idx for idx in monomer if idx != -1]
        for monomer in filled_index
    ]
    # Step 4: 映射至含氢版本
    matches_H = [
        [map_noH_to_H[idx] for idx in monomer]
        for monomer in filled_index_clean
    ]
    keypoints_poly = []
    for i in range(n_repeat):
        n1, star1, n2, star2 = keypoints
        if i==0:
            n1 = n1-1 if star1<n1 else n1        
            star1 = star1       
            n2 = n2-1 if star1<n2 else n2        
            star2 = star2-1 if star1<star2 else star2        
        n1_poly = matches_H[i][n1]
        s1_poly = matches_H[i][star1]
        n2_poly = matches_H[i][n2]
        s2_poly = matches_H[i][star2] if i!=n_repeat-1 else -1
        keypoints_poly.append([n1_poly,s1_poly,n2_poly,s2_poly])
    # Step 5: 扩展至包含H的完整 monomer
    matches_with_H = [
        expand_with_attached_H(mol_H, match)
        for match in matches_H
    ]

    return filled_index_clean, matches_H, matches_with_H,map_noH_to_H,keypoints,keypoints_poly

def expand_with_attached_H(mol_H, heavy_atom_indices):
    """
    给定一组重原子编号（来自 mol_H），返回包括其直接连接的氢原子的完整编号集合。
    """
    full_atom_indices = set(heavy_atom_indices)
    for idx in heavy_atom_indices:
        atom = mol_H.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                full_atom_indices.add(neighbor.GetIdx())
    return sorted(full_atom_indices)   
def backbone2full(mol_H,mol_noH):
    map_noH_to_H = {}  # noH idx -> H idx
    atoms_H = mol_H.GetAtoms()
    atoms_noH = mol_noH.GetAtoms()

    i_h = 0
    for i_noh, a_noh in enumerate(atoms_noH):
        while i_h < len(atoms_H):
            if atoms_H[i_h].GetAtomicNum() == a_noh.GetAtomicNum():
                map_noH_to_H[i_noh] = i_h
                i_h += 1
                break
            i_h += 1
    return map_noH_to_H

def find_connection_neighbor(mol: Chem.Mol, anchor_idx: int, monomer_atom_set: set[int]) -> int:
    """
    给定连接点 anchor（由 `[*]` 替代后连接上的原子），
    找到它在 monomer 中的连接邻居原子编号（即图中原子 4）

    参数：
    - mol: RDKit Mol 对象
    - anchor_idx: int，连接点所在的原子编号
    - monomer_atom_set: 集合，当前 monomer 的所有原子编号，用于筛选“内部原子”

    返回：
    - neighbor_idx: int，anchor 连接的 monomer 内部原子编号（唯一）
    """
    atom = mol.GetAtomWithIdx(anchor_idx)
    neighbor_indices = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() in monomer_atom_set
    ]
    
    if len(neighbor_indices) != 1:
        raise ValueError(
            f"连接点 {anchor_idx} 的 monomer 内邻居不唯一（找到 {neighbor_indices}）"
        )
    return neighbor_indices[0]
def gram_schmidt(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    使用 Gram-Schmidt 构造右手正交基 (R)
    v1 定义 x 轴方向，v2 协助定义 y 轴
    返回旋转矩阵 R: shape (3,3)
    """
    x = v1 / np.linalg.norm(v1)
    v2_proj = v2 - np.dot(v2, x) * x
    y = v2_proj / np.linalg.norm(v2_proj)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)

def compute_rigid_frame_from_three_atoms(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造局部坐标系：
    - x1, x2: 定义第一个向量 v1 = x2 - x1（前一方向）
    - x3: 当前 monomer 的 anchor，定义原点和方向 v2 = x2 - x3（后一方向）
    """
    v1 = x1 - x2
    v2 = x3 - x2  # 或 x3 - x2，只要保持一致即可
    R = gram_schmidt(v1, v2)
    t = x2
    return R, t
def get_positions_from_indices(conf, atom_indices: list[int]) -> np.ndarray:
    """
    从 conformer 中提取一组原子的坐标
    返回：np.ndarray，shape=(N, 3)
    """
    return np.array([conf.GetAtomPosition(i) for i in atom_indices])
def to_local_coords(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    将全局坐标变换为局部坐标
    X: shape (N, 3)
    R: (3, 3)
    t: (3,)
    """
    return (R.T @ (X - t).T).T

def to_global_coords(X_local: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    将局部坐标映射回全局坐标
    """
    return (R @ X_local.T).T + t


def compose_monomers(
    mols_all_H, mols_all_noH, map_noH_to_Hs, keypoints_json_path
):
    merged = Chem.RWMol()
    atom_map = {}  # (monomer_idx, old_atom_idx) -> new_atom_idx

    # 读取连接点信息
    with open(keypoints_json_path, "r") as f:
        keypoints_poly = json.load(f)

    for i in range(len(mols_all_H)):
        mol_H = mols_all_H[i]
        mol_noH = mols_all_noH[i]
        n1, star1_idx, n2, star2_idx = keypoints_poly[i]

        # 排除连接点和其氢原子
        if i == 0:
            related_Hs = expand_with_attached_H(mol_H, [star2_idx])
            exclude_idxs = [star2_idx] + related_Hs
        elif i == len(mols_all_H) - 1:
            related_Hs = expand_with_attached_H(mol_H, [star1_idx])
            exclude_idxs = [star1_idx] + related_Hs
        else:
            related_Hs = expand_with_attached_H(mol_H, [star1_idx, star2_idx])
            exclude_idxs = [star1_idx, star2_idx] + related_Hs

        # 复制原子（保留化学信息）
        for atom_idx in range(mol_H.GetNumAtoms()):
            if atom_idx in exclude_idxs:
                continue
            atom = mol_H.GetAtomWithIdx(atom_idx)
            new_atom = Chem.Atom(atom)  # 完整复制
            new_idx = merged.AddAtom(new_atom)
            atom_map[(i, atom_idx)] = new_idx

        # 复制 bond（保留芳香性）
        for bond in mol_H.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in exclude_idxs or a2 in exclude_idxs:
                continue
            new_a1 = atom_map[(i, a1)]
            new_a2 = atom_map[(i, a2)]
            merged.AddBond(new_a1, new_a2, bond.GetBondType())
            if bond.GetIsAromatic():
                merged.GetBondBetweenAtoms(new_a1, new_a2).SetIsAromatic(True)

        # 添加连接键（SINGLE）
        if i > 0:
            prev_mol_H = mols_all_H[i - 1]
            if i == 1 and star1_idx < n2:
                prev_n2_idx = map_noH_to_Hs[i - 1][n2 - 1]
            else:
                prev_n2_idx = map_noH_to_Hs[i - 1][n2]

            new_prev_idx = atom_map[(i - 1, prev_n2_idx)]
            new_curr_idx = atom_map[(i, n1)]
            merged.AddBond(new_prev_idx, new_curr_idx, rdchem.BondType.SINGLE)

    # 构建分子对象
    merged_mol = merged.GetMol()

    # 复制构象信息
    conf = Chem.Conformer(merged_mol.GetNumAtoms())
    for (i, old_idx), new_idx in atom_map.items():
        pos = mols_all_H[i].GetConformer().GetAtomPosition(old_idx)
        conf.SetAtomPosition(new_idx, pos)
    merged_mol.AddConformer(conf, assignId=True)

    # 标准化结构（避免芳香性丢失）
    Chem.SanitizeMol(merged_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return merged_mol, atom_map
def compute_rigid_transforms_for_matches(conf, matches_H, key_point_list):
    """
    计算 monomer 列表在 merged_mol 中对应的刚性变换 (R, t)

    Parameters
    ----------
    conf : RDKit.Conformer
        对应 mol_H 的构象对象
    matches_H : List[List[int]]
        每个 monomer 的原子编号匹配 (mol_H -> merged_mol)
    key_point_list : List[List[int]]
        每段连接的四个关键索引 [n1, star1, n2, star2]

    Returns
    -------
    R_arr : List[np.ndarray]
        每个 monomer 的旋转矩阵 (3x3)
    t_arr : List[np.ndarray]
        每个 monomer 的平移向量 (3,)
    """
    R_arr = []
    t_arr = []

    # 处理第一个 monomer（首段）
    match_head = matches_H[0]
    n1, star1, n2, star2 = key_point_list

    match_head_atom2 = match_head[n1 - 1] if star1 < n1 else match_head[n1]
    match_head_atom1 = match_head[n2 - 1] if star1 < n2 else match_head[n2]
    match_head_atom3 = match_head[star2 - 1] if star1 < star2 else match_head[star2]

    x1 = np.array(conf.GetAtomPosition(match_head_atom1))
    x2 = np.array(conf.GetAtomPosition(match_head_atom2))
    x3 = np.array(conf.GetAtomPosition(match_head_atom3))

    R_head, _ = compute_rigid_frame_from_three_atoms(x1, x2, x3)
    R_arr.append(R_head)
    t_arr.append(x1)  # 默认第一个 monomer 在原点

    # 处理中间 monomer（第2段开始）
    for idx in range(1, len(matches_H) - 1):
        n1, star1, n2, star2 = key_point_list

        match = matches_H[idx]
        atom1 = match[star1]
        atom2 = match[n1]
        atom3 = match[star2]

        x1 = np.array(conf.GetAtomPosition(atom1))
        x2 = np.array(conf.GetAtomPosition(atom2))
        x3 = np.array(conf.GetAtomPosition(atom3))
        R, _ = compute_rigid_frame_from_three_atoms(x1, x2, x3)
        R_arr.append(R)
        t_arr.append(x2)

    # 处理末端 monomer（尾段）
    match_tail = matches_H[-1]
    n1, star1, n2, star2 = key_point_list

    atom1 = match_tail[star1]
    atom2 = match_tail[n1]
    atom3 = match_tail[n2]

    x1 = np.array(conf.GetAtomPosition(atom1))
    x2 = np.array(conf.GetAtomPosition(atom2))
    x3 = np.array(conf.GetAtomPosition(atom3))

    R_last, _ = compute_rigid_frame_from_three_atoms(x1, x2, x3)
    R_arr.append(R_last)
    t_arr.append(x2)

    return R_arr, t_arr

def extract_submol_by_atoms(mol, atom_indices,map_noH_to_H,keypoint_list,R_arr,t_arr):
    """
    从mol中提取由atom_indices指定的子分子（包括键），保留3D构象。
    """
    # 创建原子掩码
    amap = {}
    for i, idx in enumerate(atom_indices):
        amap[idx] = i
    n1_H = keypoint_list[0]
    s1_H = keypoint_list[1]
    n2_H = keypoint_list[2]
    s2_H = keypoint_list[3]
    keypoint_list_new = [-1,-1,-1,-1]
    # 创建新分子
    emol = Chem.RWMol()
    for idx in atom_indices:
      
        atom = mol.GetAtomWithIdx(idx)
 
        new_atom = Chem.Atom(atom.GetAtomicNum())

        new_idx  = emol.AddAtom(new_atom)
        if idx == n1_H:
            keypoint_list_new[0]=new_idx
        if idx == s1_H:
            keypoint_list_new[1]=new_idx
        if idx ==n2_H:
            keypoint_list_new[2]=new_idx
        if idx ==s2_H:
            keypoint_list_new[3]=new_idx 

    bond_set = set()
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in amap and end_idx in amap:
            bond_type = bond.GetBondType()
            emol.AddBond(amap[begin_idx], amap[end_idx], bond_type)
            bond_set.add((amap[begin_idx], amap[end_idx]))

    submol = emol.GetMol()

    # 添加构象信息
    conf = mol.GetConformer()
    sub_conf = Chem.Conformer(len(atom_indices))
    for i, idx in enumerate(atom_indices):
        pos = conf.GetAtomPosition(idx)
        sub_conf.SetAtomPosition(i, pos)
    submol.RemoveAllConformers()
    submol.AddConformer(sub_conf)

    return submol,keypoint_list_new


def save_monomers_to_sdf(mol_H,map_noH_to_H, matches_with_H, output_dir,keypoint_poly,R_arr,t_arr):
    os.makedirs(output_dir, exist_ok=True)
    keypoint_list_file = os.path.join(output_dir,"keypoints.json")
    keypoint_lists = []
    for i, atom_indices in enumerate(matches_with_H):
        atom_indices = sorted(atom_indices)
        submol,keypoint_list_new = extract_submol_by_atoms(mol_H, atom_indices,map_noH_to_H,keypoint_poly[i],R_arr,t_arr)
        conf = submol.GetConformer()
        positions_global = get_positions_from_indices(conf,[i for i in range(submol.GetNumAtoms())])
        positions_local = to_local_coords(positions_global,R_arr[i],t_arr[i])
        conf.SetPositions(positions_local)        
        keypoint_lists.append(keypoint_list_new)
        writer = Chem.SDWriter(os.path.join(output_dir, f'monomer_{i}.sdf'))
        writer.write(submol)
        writer.close()
        R_arr_np = np.array(R_arr)
        t_arr_np = np.array(t_arr)
        np.save(os.path.join(output_dir,'R_arr.npy'), R_arr_np) 
        np.save(os.path.join(output_dir,'t_arr.npy'), t_arr_np) 
    with open(keypoint_list_file,"w") as f:
        json.dump(keypoint_lists, f)
    