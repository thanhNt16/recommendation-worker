import logging
import json
import pika
from threading import Thread

class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret

class Publisher:
    def __init__(self, host, virtual_host, username, password, queue):
        self._params = pika.connection.ConnectionParameters(
            host=host,
            virtual_host=virtual_host,
            credentials=pika.credentials.PlainCredentials(username, password),
            heartbeat=3600)
        self._conn = None
        self._channel = None
        self.QUEUE = queue

    def connect(self):
        if not self._conn or self._conn.is_closed:
            self._conn = pika.BlockingConnection(self._params)
            self._channel = self._conn.channel()
            # self._channel.exchange_declare(exchange=self.EXCHANGE,
            #                                type=self.TYPE)

    def _consume(self, queue, callback):
        if (self._channel != None):
            try:
                self._channel.basic_consume(queue=queue,
                                            auto_ack=False,
                                            on_message_callback=callback)
                print(' [*] Waiting for messages. To exit press CTRL+C')
        
                thread = PropagatingThread(target=self._channel.start_consuming)
                thread.start()
                thread.join()
                # print(123)
            except Exception as e:
                self.connect()
                self.consume(queue, callback)
        else:
            self.connect()
            self.consume(queue, callback)

    def consume(self, queue, callback):
        """Publish msg, reconnecting if necessary."""

        try:
            self._consume(queue, callback)
        except Exception as e:
            logging.debug('reconnecting to queue')
            self.connect()
            self._consume(queue, callback)

    def _publish(self, msg):
        self._channel.basic_publish('', self.QUEUE, msg)
        logging.debug('message sent: %s', msg)

    def publish(self, msg):
        """Publish msg, reconnecting if necessary."""

        try:
            self._publish(msg)
        except (Exception) as e:
            logging.debug('reconnecting to queue')
            self.connect()
            self._publish(msg)

    def close(self):
        if self._conn and self._conn.is_open:
            logging.debug('closing queue connection')
            self._conn.close()