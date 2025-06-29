import json
import argparse
from multiprocessing import Process
import os
import shutil
from utils import pika_utils
from utils.pysoftk_utils import psmiles2ChainPoly
from utils.pika_utils import RMQConfig
from utils.log_utils import node_print,worker_print
import time
import pika

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
        with open(os.path.join(save_folder,"psmiles.txt"),"w") as f:
            f.write(msg["psmiles"].replace("Pt","*"))
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
        ch.basic_ack(delivery_tag=method.delivery_tag)
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
        ch.basic_ack(delivery_tag=method.delivery_tag)
        node_print(config.node,f"Task {task_id}-{msg['psmiles']} has been redirected to failed queue: {config.failed_queue_name}")


def chain_builder_worker_main(worker_id: int, config_path):
    config = ChainBuilderConfig(config_path)
    worker_print(config.node, worker_id, f"Chain builder worker started! cwd:{os.getcwd()}")
    rmq_config = RMQConfig(config_path)

    def create_channel_and_consume():
        """建立新连接并绑定消费逻辑"""
        rabbitmq_utils = pika_utils.RabbitMQUtils(rmq_config)
        channel = rabbitmq_utils.connection.channel()
        channel.queue_declare(queue=config.queue_name, durable=True)
        channel.basic_qos(prefetch_count=1)

        # 定义消息回调
        def callback(ch, method, properties, body):
            on_message(ch, method, properties, body, config, rabbitmq_utils)
        channel.basic_consume(queue=config.queue_name, on_message_callback=callback, auto_ack=False)
        return channel

    while True:
        try:
            channel = create_channel_and_consume()
            worker_print(config.node, worker_id, f'Waiting for messages in queue: {config.queue_name}. Press CTRL+C to exit.')
            channel.start_consuming()
        except pika.exceptions.AMQPError as e:
            worker_print(config.node, worker_id, f"AMQP error: {e}, attempting to reconnect...")
            time.sleep(5)
        except Exception as e:
            worker_print(config.node, worker_id, f"Unexpected error: {e}, retrying in 5 seconds...")
            time.sleep(5)

def start_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file', default='configs/config.json')
    parser.add_argument('--workers', help='Num for workers', default=1)
    args = parser.parse_args()
    processes = []
    for i in range(int(args.workers)):
        p = Process(target=chain_builder_worker_main, args=(i,args.config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    start_process()
