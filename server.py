import json
import logging
import socket
import time
from tracemalloc import start
from typing import Tuple
import config
import numpy as np
import threading
import utils

import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)

model = load_model(config.MODEL_FOLDER + "/model.h5")

@tf.function
def predict(args) -> Tuple[list[float], float]:
	return model(args)


class ServerSocket:
	def __init__(self, host, port):
		self.host = host
		self.port = port
		# first prediction
		test_data = np.random.choice(a=[False, True], size=(1, *config.INPUT_SHAPE), p=[0, 1])
		p, v = predict(test_data)
		del test_data, p, v


	def start(self):
		logging.info("Starting server...")
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.sock.bind((self.host, self.port))
		# listen for incoming connections, queue up to 24 requests
		self.sock.listen(24)
		logging.info("Server started.")
		try:
			while True:
				self.accept()
				logging.info(f"Current thread count: {threading.active_count()}.")
		except KeyboardInterrupt:
			self.stop()
		except Exception as e:
			logging.debug(f"Error: {e}")
			self.sock.close()

	def accept(self):
		logging.info("Waiting for client...")
		self.client, address = self.sock.accept()
		logging.info(f"Client connected from {address}")
		clh = ClientHandler(self.client, address)
		# start new thread to handle client
		clh.start()

	def stop(self):
		logging.info("Stopping server...")
		self.sock.close()
		logging.info("Server stopped.")


class ClientHandler(threading.Thread):
	def __init__(self, sock: socket.socket, address: Tuple[str, int]):
		super().__init__()
		self.BUFFER_SIZE = config.SOCKET_BUFFER_SIZE
		self.sock = sock
		self.address = address

	def run(self):
		print(f"ClientHandler started.")
		while True:
			data = self.receive()
			if data is None or len(data) == 0:
				self.close()
				break
			data = np.array(np.frombuffer(data, dtype=bool))
			data = data.reshape(1, *config.INPUT_SHAPE)
			# make prediction
			p, v = predict(data)
			p, v = p[0].numpy().tolist(), float(v[0][0])
			response = json.dumps({"prediction": p, "value": v})
			self.send(f"{len(response):010d}".encode('ascii'))
			self.send(response.encode('ascii'))

	def receive(self):
		data = None
		try:
			data = utils.recvall(self.sock)
			if len(data) != 1216:
				data = None
				raise ValueError("Invalid data length, closing socket")
		except ConnectionResetError:
			logging.warning(f"Connection reset by peer. Client IP: {str(self.address[0])}:{str(self.address[1])}")
		except ValueError as e:
			logging.warning(e)
		return data

	def send(self, data):
		logging.debug("Sending data...")
		self.sock.send(data)
		logging.debug("Data sent.")

	def close(self):
		logging.info("Closing connection...")
		self.sock.close()
		logging.info("Connection closed.")
	

if __name__ == "__main__":
	s = ServerSocket(config.SOCKET_HOST, config.SOCKET_PORT)
	s.start()
	