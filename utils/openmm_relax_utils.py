from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openmm.app import ForceField, Simulation, Modeller, NoCutoff, DCDReporter, StateDataReporter
from openmm import unit, Platform, LangevinMiddleIntegrator
from openff.toolkit.utils import RDKitToolkitWrapper
from openff.units.openmm import to_openmm
from openff.units import unit as off_unit
from openmmml import MLPotential

import numpy as np
import sys
import time
from datetime import datetime

def run_md_simulation(
    input_file: str = "chain.sdf",
    n_steps: int = 5_000_000,
    forcefield_name: str = "gaff-2.11",
    platform_name: str = "CUDA",
    precision:str = "mixed",
    temperature_kelvin: float = 298.0,
    timestep_fs: float = 1.0,
    output_trajectory: str = "trajectory.dcd",
    output_sdf: str = "relaxed_chain.sdf"
):
    # 1. 加载分子
    mol = Molecule.from_file(input_file, file_format="sdf")

    # 2. 分配 Gasteiger 电荷
    toolkit = RDKitToolkitWrapper()
    mol.assign_partial_charges("gasteiger", toolkit_registry=toolkit)

    # 3. 准备 GAFF 力场模板
    gaff = GAFFTemplateGenerator(
        molecules=[mol],
        force_field=forcefield_name,
        charge_from_molecules=True
    )

    # 4. 构建力场对象
    forcefield = ForceField()
    forcefield.registerTemplateGenerator(gaff.generator)

    # 5. 构造 Modeller
    topology = mol.to_topology().to_openmm()
    positions = to_openmm(mol.conformers[0])
    modeller = Modeller(topology, positions)

    # 6. 创建非周期系统
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None
    )

    # 7. 选择平台
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {"Precision": precision} if platform_name in ["CUDA", "OpenCL"] else {}
    except Exception as e:
        print(f"Platform {platform_name} not available: {e}")
        platform = Platform.getPlatformByName("CPU")
        properties = {}
        print("Fallback to CPU platform")

    # 8. 设置积分器与模拟器
    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtoseconds
    )
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    # 9. 最小化能量
    simulation.minimizeEnergy()

    # 10. 设置初速度并添加报告器
    simulation.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)
    simulation.reporters.append(DCDReporter(output_trajectory, 1000))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 1000, step=True, temperature=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True, progress=True, remainingTime=True,
        speed=True, totalSteps=n_steps, separator='\t'))

    # 11. 执行模拟
    print("Starting simulation...")
    start_time = time.time()
    simulation.step(n_steps)
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds "
          f"({elapsed_time / 60:.2f} minutes, {elapsed_time / 3600:.2f} hours)")

    # 12. 提取最终构象并保存
    state = simulation.context.getState(getPositions=True)
    relaxed_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    mol._conformers = [relaxed_positions * off_unit.angstrom]
    mol.to_file(output_sdf, file_format="sdf")
    print(f"Relaxed structure saved to {output_sdf}")

def run_md_simulation_ml(
    input_file: str = "chain.sdf",
    n_steps: int = 5_000_000,
    forcefield_name: str = "ani2x",
    platform_name: str = "CUDA",
    precision:str = "mixed",
    temperature_kelvin: float = 298.0,
    timestep_fs: float = 1.0,
    output_trajectory: str = "trajectory.dcd",
    output_sdf: str = "relaxed_chain.sdf"
):
    # 1. 加载分子
    mol = Molecule.from_file(input_file, file_format="sdf")

    # 2. 分配 Gasteiger 电荷
    toolkit = RDKitToolkitWrapper()
    mol.assign_partial_charges("gasteiger", toolkit_registry=toolkit)

    # 3. 准备 GAFF 力场模板
    # gaff = GAFFTemplateGenerator(
    #     molecules=[mol],
    #     force_field=forcefield_name,
    #     charge_from_molecules=True
    # )

    # 4. 构建力场对象
    forcefield = MLPotential('ani2x')
    # forcefield.registerTemplateGenerator(gaff.generator)
    # 5. 构造 Modeller
    topology = mol.to_topology().to_openmm()
    positions = to_openmm(mol.conformers[0])
    modeller = Modeller(topology, positions)

    # 6. 创建非周期系统
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None
    )

    # 7. 选择平台
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {"Precision": precision} if platform_name in ["CUDA", "OpenCL"] else {}
    except Exception as e:
        print(f"Platform {platform_name} not available: {e}")
        platform = Platform.getPlatformByName("CPU")
        properties = {}
        print("Fallback to CPU platform")

    # 8. 设置积分器与模拟器
    integrator = LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtoseconds
    )
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    # 9. 最小化能量
    simulation.minimizeEnergy()

    # 10. 设置初速度并添加报告器
    simulation.context.setVelocitiesToTemperature(temperature_kelvin * unit.kelvin)
    simulation.reporters.append(DCDReporter(output_trajectory, 1000))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 1000, step=True, temperature=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True, progress=True, remainingTime=True,
        speed=True, totalSteps=n_steps, separator='\t'))

    # 11. 执行模拟
    print("Starting simulation...")
    start_time = time.time()
    simulation.step(n_steps)
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds "
          f"({elapsed_time / 60:.2f} minutes, {elapsed_time / 3600:.2f} hours)")

    # 12. 提取最终构象并保存
    state = simulation.context.getState(getPositions=True)
    relaxed_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    mol._conformers = [relaxed_positions * off_unit.angstrom]
    mol.to_file(output_sdf, file_format="sdf")
    print(f"Relaxed structure saved to {output_sdf}")
if __name__ == "__main__":
    run_md_simulation(
    input_file="results/task_20250628_001",
    n_steps=250_000,
    platform_name="CUDA",
    precision = "mixed",
    temperature_kelvin=289,
    forcefield_name="gaff-2.11",
    timestep_fs=2.0,
    output_sdf="relaxed_chain.sdf",
    output_trajectory="trajectory.dcd",
)
#     run_md_simulation_ml(
#     input_file="chain.sdf",
#     n_steps=5_000_000,
#     platform_name="CUDA",
#     precision = "mixed",
#     temperature_kelvin=289,
#     forcefield_name="ani2x",
#     timestep_fs=2.0,
#     output_sdf="relaxed_chain.sdf",
#     output_trajectory="trajectory.dcd",
# )