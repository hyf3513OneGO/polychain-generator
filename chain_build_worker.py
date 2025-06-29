import json
import argparse
import os
import shutil
from utils import pika_utils
from utils.pysoftk_utils import psmiles2ChainPoly
from utils.pika_utils import RMQConfig
from utils.log_utils import node_print
import time

class ChainBuilderConfig:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            self.node = self.config["chainBuilder"]["node"]
            self.queue_name = self.config["chainBuilder"]["queue_name"]
            self.failed_queue_name = self.config["chainBuilder"].get("failed_queue_name", "chainBuilder_failed")
            self.relax_queue_name = self.config["chainBuilder"]["relax_queue_name"]


def on_message(ch, method, properties, body, config: ChainBuilderConfig, rabbitmq_utils):
    msg_body = body.decode()
    msg = json.loads(msg_body)
    task_id = msg["id"]
    save_folder = os.path.join("results", task_id)

    start_time = time.time()
    node_print(config.node,f"Chain Build Task <{task_id}> started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    try:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        
        psmiles2ChainPoly(
            psmiles=msg["psmiles"],
            n_repeat=msg["n_repeat"],
            save_path=os.path.join(save_folder, "init_chain.sdf"),
            relaxation_steps=msg["relaxation_steps"],
            force_field=msg["force_field"],
        )

        end_time = time.time()
        elapsed = end_time - start_time
        relax_body = msg["relaxation"]
        relax_body ["id"] = msg["id"]
        relax_body ["psmiles"] = msg["psmiles"]
        relax_body = json.dumps(relax_body).encode()
        relax_channel = rabbitmq_utils.connection.channel()
        relax_channel.queue_declare(queue=config.relax_queue_name, durable=True)
        relax_channel.basic_publish(
            exchange='',
            routing_key=config.relax_queue_name,
            body=relax_body,
            properties=properties
        )
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


def chain_builder_worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', default='configs/config.json')
    args = parser.parse_args()

    config = ChainBuilderConfig(args.config)
    node_print(config.node,f"Chain builder worker started! cwd:{os.getcwd()}")

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
    chain_builder_worker_main()
