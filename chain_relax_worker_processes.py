import json
import os
import shutil
import time
import argparse
import asyncio
from aio_pika import connect_robust, IncomingMessage, Message

from utils.pika_utils import RMQConfig
from utils.log_utils import node_print, worker_print
from utils.openmm_relax_utils import run_md_simulation


class ChainRelaxConfig:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            config = json.load(f)
            self.node = config["chainRelaxation"]["node"]
            self.queue_name = config["chainRelaxation"]["queue_name"]
            self.failed_queue_name = config["chainRelaxation"].get("failed_queue_name", "chainRelaxation_failed")

def process_relaxation_sync(msg: dict):
    task_id = msg["id"]
    prefix_folder = msg["prefix"]
    save_folder = os.path.join("results", prefix_folder, task_id)

    if not os.path.exists(save_folder):
        raise FileNotFoundError("Init Chain conformation not found")

    run_md_simulation(
        input_file=os.path.join(save_folder, "init_chain.sdf"),
        n_steps=int(msg["n_steps"]),
        forcefield_name=msg["forcefield"],
        platform_name=msg["platform"],
        precision=msg["precision"],
        temperature_kelvin=float(msg["temperature_kelvin"]),
        timestep_fs=float(msg["timestep_fs"]),
        output_trajectory=os.path.join(save_folder, "trajectory.dcd"),
        output_sdf=os.path.join(save_folder, "relaxed_chain.sdf")
    )
    return task_id, msg

def run_task_in_subprocess(msg):
    try:
        return process_relaxation_sync(msg)
    except Exception as e:
        return ("error", msg, str(e))


async def handle_message(message: IncomingMessage, config: ChainRelaxConfig, channel):
    async with message.process():
        try:
            msg = json.loads(message.body.decode())
            task_id = msg["id"]
            node_print(config.node, f"Chain Relax Task <{task_id}> started")

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_task_in_subprocess, msg)

            if result[0] == "error":
                raise RuntimeError(result[2])

            task_id, msg = result
            node_print(config.node, f"Task {task_id} completed successfully.")

        except Exception as e:
            try:
                msg = json.loads(message.body.decode())
                task_id = msg.get("id", "unknown")
                save_folder = os.path.join("results", msg.get("prefix", ""), task_id)
                if os.path.exists(save_folder):
                    shutil.rmtree(save_folder)
            except Exception:
                pass
            await channel.default_exchange.publish(
                Message(body=message.body, headers=message.headers, content_type="application/json"),
                routing_key=config.failed_queue_name
            )
            node_print(config.node, f"Task {task_id} failed with error: {e} and sent to failed queue.")


async def worker_main(worker_id: int, config_path: str):
    config = ChainRelaxConfig(config_path)
    rmq_config = RMQConfig(config_path)
    worker_print(config.node, worker_id, f"[Worker {worker_id}] Starting ChainRelax with aio-pika")

    connection = await connect_robust(
        host=rmq_config.host,
        port=rmq_config.port,
        login=rmq_config.username,
        password=rmq_config.password,
        virtualhost=rmq_config.virtual_host,
        heartbeat=30
    )
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(config.queue_name, durable=True)

    await queue.consume(lambda msg: handle_message(msg, config, channel), no_ack=False)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await connection.close()
        worker_print(config.node, worker_id, f"[Worker {worker_id}] Connection closed.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', default='configs/config.json')
    parser.add_argument('--workers', help='Num for workers', default=16)
    args = parser.parse_args()

    tasks = [asyncio.create_task(worker_main(i, args.config)) for i in range(int(args.workers))]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested")