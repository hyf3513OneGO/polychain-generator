from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField, Simulation, Modeller, NoCutoff, DCDReporter, StateDataReporter
from openmm import unit, Platform, LangevinMiddleIntegrator
from openff.toolkit.utils import RDKitToolkitWrapper
from openff.units.openmm import to_openmm
from openff.units import unit as off_unit

import numpy as np
import sys

# === 1. 从 SDF 文件加载聚合物链 ===
mol = Molecule.from_file("chain.sdf", file_format="sdf")

# === 2. 使用 Gasteiger 电荷（避免调用 AM1-BCC）===
toolkit = RDKitToolkitWrapper()
mol.assign_partial_charges("gasteiger", toolkit_registry=toolkit)

# === 3. 初始化 GAFF 力场模板 ===
gaff = GAFFTemplateGenerator(
    molecules=[mol],
    force_field='gaff-2.11',
    charge_from_molecules=True
)

# === 4. 创建 ForceField 并注册 GAFF 模板 ===
forcefield = ForceField()
forcefield.registerTemplateGenerator(gaff.generator)

# === 5. 准备 OpenMM 拓扑和位置 ===
topology = mol.to_topology().to_openmm()
positions = to_openmm(mol.conformers[0])
modeller = Modeller(topology, positions)

# === 6. 创建非周期体系的 System ===
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=NoCutoff,
    constraints=None
)

# === 7. 初始化模拟器，使用 NVT（恒温恒体积）设置 ===
integrator = LangevinMiddleIntegrator(
    298 * unit.kelvin,
    1.0 / unit.picosecond,
    0.001 * unit.picoseconds  # 1 fs
)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

# === 8. 最小化能量 ===
simulation.minimizeEnergy()

# === 9. 设置初始速度并运行 5 ns 模拟 ===
simulation.context.setVelocitiesToTemperature(298 * unit.kelvin)

# 添加轨迹输出（可选）
simulation.reporters.append(DCDReporter("trajectory.dcd", 1000))
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
    temperature=True, potentialEnergy=True,kineticEnergy=True,totalEnergy=True, progress=True, remainingTime=True,
    speed=True, totalSteps=5_000_000, separator='\t'))

# === 10. 运行模拟（5 ns）===
simulation.step(5_000_000)

# === 11. 提取最终构象并保存 ===
state = simulation.context.getState(getPositions=True)
relaxed_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
mol._conformers = [relaxed_positions * off_unit.angstrom]
mol.to_file("relaxed_chain.sdf", file_format="sdf")

print("Conformation relaxed and saved to relaxed_chain.sdf")
