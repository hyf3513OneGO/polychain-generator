from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm.app import *
from openmm import *
from openmm.unit import *
import sys

def smiles_to_relaxed_structure(smiles, output_pdb="relaxed.pdb", n_steps=10000):
    # Step 1: 生成 3D 构象（RDKit）
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    # 保存为 mol file
    Chem.MolToMolFile(mol, "temp.mol")

    # Step 2: OpenFF 分子对象
    off_mol = Molecule.from_file("temp.mol")
    off_mol.generate_conformers(n_conformers=1)

    # Step 3: 创建力场系统
    forcefield = ForceField('amber14-all.xml')
    topology = off_mol.to_topology()
    system = forcefield.createSystem(topology,nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)

    # Step 4: OpenMM 模拟设置
    pdbfile = PDBFile(off_mol.to_file("temp.pdb", file_format="PDB"))
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(pdbfile.topology, system, integrator)
    simulation.context.setPositions(pdbfile.positions)

    # Step 5: 能量最小化
    print("开始能量最小化...")
    simulation.minimizeEnergy()

    # Step 6: 可选短程模拟（用于放松结构）
    print(f"运行 {n_steps} 步动力学以进一步弛豫...")
    simulation.context.setVelocitiesToTemperature(300 * kelvin)
    simulation.step(n_steps)

    # Step 7: 输出结果
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(output_pdb, "w"))
    print(f"结构已写入 {output_pdb}")

if __name__ == "__main__":
    smiles = "CC(C)CC(C)C"  # 示例：支链烷烃
    smiles_to_relaxed_structure(smiles)
