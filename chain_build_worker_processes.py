import json
import argparse
import os
import shutil
import asyncio
from concurrent.futures import ProcessPoolExecutor
from aio_pika import connect_robust, IncomingMessage, Message

from utils.pysoftk_utils import psmiles2ChainPoly
from utils.pika_utils import RMQConfig
from utils.log_utils import node_print, worker_print


class ChainBuilderConfig:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            config = json.load(f)
            self.node = config["chainBuilder"]["node"]
            self.queue_name = config["chainBuilder"]["queue_name"]
            self.failed_queue_name = config["chainBuilder"].get("failed_queue_name", "chainBuilder_failed")
            self.relax_queue_name = config["chainBuilder"]["relax_queue_name"]


def process_chain_build_sync(msg: dict):
    task_id = msg["id"]
    prefix_folder = msg["prefix"]
    save_folder = os.path.join("results",prefix_folder, task_id)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "psmiles.txt"), "w") as f:
        f.write(msg["psmiles"].replace("Pt", "*"))
    psmiles2ChainPoly(
        psmiles=msg["psmiles"],
        n_repeat=msg["n_repeat"],
        save_path=os.path.join(save_folder, "init_chain.sdf"),
        relaxation_steps=msg["relaxation_steps"],
        force_field=msg["force_field"],
    )
    return task_id, msg


async def handle_message(message: IncomingMessage, config: ChainBuilderConfig, channel, pool: ProcessPoolExecutor):
    async with message.process():
        try:
            msg = json.loads(message.body.decode())
            task_id = msg["id"]
            node_print(config.node, f"Chain Build Task <{task_id}> started")

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(pool, process_chain_build_sync, msg)
            task_id, msg = result
            while not os.path.exists(os.path.join("results", msg["prefix"], task_id, "init_chain.sdf")):
                node_print(config.node, f"Waiting for chain build result for task {task_id}...")
                await asyncio.sleep(1)
            relax_body = msg["relaxation"]
            relax_body["id"] = msg["id"]
            relax_body["prefix"] = msg["prefix"]
            relax_body["psmiles"] = msg["psmiles"]
            body = json.dumps(relax_body).encode()
            await channel.default_exchange.publish(
                Message(body=body, headers=message.headers, content_type="application/json"),
                routing_key=config.relax_queue_name
            )
            node_print(config.node, f"Task {msg['id']} completed and forwarded to relax queue.")

        except Exception as e:
            msg = json.loads(message.body.decode())
            task_id = msg.get("id", "unknown")
            save_folder = os.path.join("results", task_id)
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
            await channel.default_exchange.publish(
                Message(body=message.body, headers=message.headers, content_type="application/json"),
                routing_key=config.failed_queue_name
            )
            node_print(config.node, f"Task {task_id} failed with error: {e} and sent to failed queue.")


async def worker_main(worker_id: int, config_path: str, pool: ProcessPoolExecutor):
    config = ChainBuilderConfig(config_path)
    rmq_config = RMQConfig(config_path)
    worker_print(config.node, worker_id, f"[Worker {worker_id}] Starting with aio-pika")

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

    await queue.consume(lambda msg: handle_message(msg, config, channel, pool), no_ack=False)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await connection.close()
        worker_print(config.node, worker_id, f"[Worker {worker_id}] Connection closed.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', default='configs/config.json')
    parser.add_argument('--workers', help='Num for workers', default=15)
    args = parser.parse_args()

    pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    tasks = [asyncio.create_task(worker_main(i, args.config, pool)) for i in range(int(args.workers))]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested")
