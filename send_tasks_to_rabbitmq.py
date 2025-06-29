import json
import argparse
from utils import pika_utils
from utils.pika_utils import RMQConfig


def send_tasks_to_rabbitmq(text_file_path, config_path):
    # 加载配置
    rmq_config = RMQConfig(config_path)
    rabbitmq_utils = pika_utils.RabbitMQUtils(rmq_config)
    channel = rabbitmq_utils.connection.channel()

    # 读取配置中的队列名
    with open(config_path, "r") as f:
        config = json.load(f)
        queue_name = config["chainBuilder"]["queue_name"]

    # 基础消息模板
    base_message = {
        "id": "",
        "psmiles": "",
        "n_repeat": 5,
        "relaxation_steps": 350,
        "force_field": "MMFF",
        "relaxation": {
            "n_steps": 5000000,
            "platform": "CUDA",
            "precision": "mixed",
            "temperature_kelvin": 289,
            "forcefield": "gaff-2.11",
            "timestep_fs": 2.0
        }
    }

    # 读取文本文件
    with open(text_file_path, "r") as f:
        for idx, line in enumerate(f, start=1):
            psmiles = line.strip()
            if psmiles:
                # 更新消息内容
                message = base_message.copy()
                message["id"] = f"task_{idx}"
                message["psmiles"] = psmiles

                # 发送消息到队列
                channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=json.dumps(message)
                )
                print(f"Sent task {message['id']} with psmiles: {psmiles}")

    # 关闭连接
    rabbitmq_utils.connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send tasks to RabbitMQ queue with psmiles from a text file.')
    parser.add_argument('--text_file', required=True, help='Path to the text file containing psmiles')
    parser.add_argument('--config', default='configs/config.json', help='Path to the configuration file')
    args = parser.parse_args()

    send_tasks_to_rabbitmq(args.text_file, args.config)