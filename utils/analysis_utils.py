from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
def estimate_unit_size_from_psmiles(psmiles: str) -> int:
    """通过 PSMILES 估计单体原子数，包括连接位点"""
    mol = Chem.MolFromSmiles(psmiles)
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()

def extract_atom_coords_from_sdf(sdf_file: str) -> np.ndarray:
    """从 SDF 文件中提取原子坐标数组"""
    mol = Chem.MolFromMolFile(sdf_file, removeHs=False)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

def extract_repeating_units(coords: np.ndarray, unit_size: int, shared_atoms: int = 2):
    """从聚合物坐标中提取所有重复单体的原子坐标"""
    stride = unit_size - shared_atoms
    units = []
    for i in range(0, len(coords) - unit_size + 1, stride):
        unit_coords = coords[i:i+unit_size]
        units.append(unit_coords)
    return units

def gram_schmidt(v1, v2):
    """构造局部坐标系的旋转矩阵 R"""
    x = v1 / np.linalg.norm(v1)
    z = np.cross(v1, v2)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z]).T

def extract_frame(unit_coords: np.ndarray):
    """从单体原子中提取局部刚体变换 (R, t)"""
    v1 = unit_coords[1] - unit_coords[0]
    v2 = unit_coords[2] - unit_coords[0]
    R = gram_schmidt(v1, v2)
    t = unit_coords[0]
    return R, t

def normalize_unit(unit_coords: np.ndarray):
    """将单体构象标准化至本地坐标系（剥离 R 和 t）"""
    R, t = extract_frame(unit_coords)
    normalized = (unit_coords - t) @ R.T
    return normalized, R, t

# === 主流程 ===
def main(result_folder:str):
    psmiles = ""
    sdf_file = os.path.join(result_folder,"relaxed_chain.sdf")
    smiles_file = os.path.join(result_folder,"psmiles.txt")
    with open(smiles_file,"r") as f:
        psmiles = f.read().strip()
    
    unit_size = estimate_unit_size_from_psmiles(psmiles)
    coords = extract_atom_coords_from_sdf(sdf_file)
    units = extract_repeating_units(coords, unit_size, shared_atoms=2)

    normalized_units = []
    transforms = []

    for unit in units:
        norm, R, t = normalize_unit(unit)
        normalized_units.append(norm)
        transforms.append((R, t))
if __name__ =="__main__":
    print(os.getcwd())
    main("results/task_1_repeat-5_25000_2.0_298.0k_gaff-2.11")
