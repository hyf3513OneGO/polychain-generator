import pika
import json



class RMQConfig:
    def __init__(self,config_path:str):
        self.config_path = config_path
        with open(self.config_path,"r") as f:
            self.config = json.load(f)
            self.host = self.config["rmq"]["host"]
            self.port = self.config["rmq"]["port"]
            self.username = self.config["rmq"]["username"]
            self.password = self.config["rmq"]["password"]
            self.virtual_host = self.config["rmq"]["virtual_host"]
            
class RabbitMQUtils:
    def __init__(self, rmq_config:RMQConfig, max_retries=3, retry_delay=5):
        """
        初始化RabbitMQ连接

        :param host: RabbitMQ主机地址
        :param port: RabbitMQ端口
        :param username: 用户名
        :param password: 密码
        :param virtual_host: 虚拟主机
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔时间(秒)
        """
        credentials = pika.PlainCredentials(rmq_config.username,rmq_config.password)
        self.connection_params = pika.ConnectionParameters(
            host=rmq_config.host,
            port=rmq_config.port,
            virtual_host=rmq_config.virtual_host,
            credentials=credentials,
            heartbeat=600,                    # 设置心跳（秒），例如 1000 分钟
            blocked_connection_timeout=60    # 被阻塞连接最大等待时长（秒）
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection = self._get_connection()

    def _get_connection(self):
        """获取RabbitMQ连接，带重连机制"""
        for attempt in range(self.max_retries):
            try:
                return pika.BlockingConnection(self.connection_params)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"连接失败，第 {attempt + 1} 次重试...错误信息: {e}")
                    import time
                    time.sleep(self.retry_delay)
                else:
                    print(f"连接失败，已达到最大重试次数 {self.max_retries}。错误信息: {e}")
                    raise

    def create_queue(self, queue_name, durable=False):
        """
        创建队列

        :param queue_name: 队列名称
        :param durable: 是否持久化队列
        :return: 队列声明结果
        """
        try:
            channel = self.connection.channel()
            result = channel.queue_declare(queue=queue_name, durable=durable)
            return result
        except Exception as e:
            print(f"创建队列失败: {e}")
            return None

    def delete_queue(self, queue_name):
        """
        删除队列

        :param queue_name: 队列名称
        :return: 是否删除成功
        """
        try:
            channel = self.connection.channel()
            channel.queue_delete(queue=queue_name)
            return True
        except Exception as e:
            print(f"删除队列失败: {e}")
            return False

    def update_queue(self, queue_name, new_durable=None):
        """
        更新队列属性（当前仅支持更新持久化属性）

        :param queue_name: 队列名称
        :param new_durable: 新的持久化状态
        :return: 是否更新成功
        """
        try:
            if new_durable is None:
                print("未提供更新参数")
                return False
            channel = self.connection.channel()
            channel.queue_declare(queue=queue_name, durable=new_durable)
            return True
        except Exception as e:
            print(f"更新队列失败: {e}")
            return False

    def __del__(self):
        """
        析构函数，关闭连接
        """
        if self.connection and not self.connection.is_closed:
            self.connection.close()

if __name__ == "__main__":
    rmq_config = RMQConfig("configs/config.json")
    rabbitmq_utils = RabbitMQUtils(rmq_config)
    rabbitmq_utils.create_queue("test_queue")
    def on_message(channel, method_frame, header_frame, body):
        print(method_frame.delivery_tag)
        print(body)
        print()
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)

    channel = rabbitmq_utils.connection.channel()
    channel.basic_consume("test_queue",on_message)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()