import json
from utils.openmm_relax_utils import run_md_simulation
import os
import shutil
from utils import pika_utils
from utils.pika_utils import RMQConfig
from utils.log_utils import node_print
import argparse
import time

class ChainRelaxConfig:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            self.node = self.config["chainRelaxation"]["node"]
            self.queue_name = self.config["chainRelaxation"]["queue_name"]
            self.failed_queue_name = self.config["chainRelaxation"].get("failed_queue_name", "chainBuilder_failed")
def on_message(ch, method, properties, body, config: ChainRelaxConfig, rabbitmq_utils):
    msg_body = body.decode()
    msg = json.loads(msg_body)
    task_id = msg["id"]
    save_folder = os.path.join("results", task_id)
    
    start_time = time.time()
    node_print(config.node,f"Chain Relax Task <{task_id}> started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    try:
        if not os.path.exists(save_folder):
            raise f"Init Chain conformation not found"
        
        run_md_simulation(
            input_file=os.path.join(save_folder, "init_chain.sdf"),
            n_steps=msg["n_steps"],
            forcefield_name=msg["forcefield"],
            platform_name=msg["platform"],
            precision=msg["precision"],
            temperature_kelvin=msg["temperature_kelvin"],
            timestep_fs=msg["timestep_fs"],
            output_trajectory=os.path.join(save_folder, "trajectory.dcd"),
            output_sdf=os.path.join(save_folder, "relaxed_chain.sdf")
        )

        end_time = time.time()
        elapsed = end_time - start_time
        node_print(config.node,f"Task {task_id} finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, elapsed time: {elapsed:.2f} seconds")

    except Exception as e:
        node_print(config.node,f"Task {task_id}-{msg['psmiles']} failed with error: {e}")
        if os.path.exists(save_folder) and os.path.isdir(save_folder):
            shutil.rmtree(save_folder)

        failed_channel = rabbitmq_utils.connection.channel()
        failed_channel.queue_declare(queue=config.failed_queue_name, durable=True)
        failed_channel.basic_publish(
            exchange='',
            routing_key=config.failed_queue_name,
            body=body,
            properties=properties
        )
        node_print(config.node,f"Task {task_id}-{msg['psmiles']} has been redirected to failed queue: {config.failed_queue_name}")
def chain_relax_worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', default='configs/config.json')
    args = parser.parse_args()
    config = ChainRelaxConfig(args.config)
    node_print(config.node,f"Chain relax worker started! cwd:{os.getcwd()}")

    rmq_config = RMQConfig(args.config)
    rabbitmq_utils = pika_utils.RabbitMQUtils(rmq_config)
    channel = rabbitmq_utils.connection.channel()
    channel.queue_declare(queue=config.queue_name, durable=True)

    # Wrap the message handler with context
    def callback(ch, method, properties, body):
        on_message(ch, method, properties, body, config, rabbitmq_utils)

    channel.basic_consume(queue=config.queue_name, on_message_callback=callback, auto_ack=True)
    node_print(config.node,f'Waiting for messages in queue: {config.queue_name}. Press CTRL+C to exit.')
    channel.start_consuming()

if __name__ == "__main__":
    chain_relax_worker_main()


