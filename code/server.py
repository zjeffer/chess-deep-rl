import json
import logging
import os
import socket
import time
from tracemalloc import start
from typing import Tuple
import config
import numpy as np
import threading
import utils

from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format=' %(message)s')

model = load_model(config.MODEL_FOLDER + "/model-2022-04-18_00:29:41-from-12-april.h5")

@tf.function(experimental_follow_type_hints=True)
def predict(args: tf.Tensor) -> Tuple[list[tf.float32], list[list[tf.float32]]]:
	return model(args)


class ServerSocket:
	def __init__(self, host, port):
		"""
		The server object listens to connections and creates client handlers
		for every client (multi-threaded).

		It receives inputs from the clients and returns the predictions to the correct client.
		"""
		self.host = host
		self.port = port
		# first prediction
		test_data = np.random.choice(a=[False, True], size=(1, *config.INPUT_SHAPE), p=[0, 1])
		tf.convert_to_tensor(test_data, dtype=tf.bool)
		p, v = predict(test_data)
		del test_data, p, v


	def start(self):
		"""
		Start the server and listen for connections.
		"""
		logging.info(f"Starting server on {self.host}:{self.port}...")
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		self.sock.bind((self.host, self.port))
		# listen for incoming connections, queue up to 24 requests
		self.sock.listen(24)
		logging.info(f"Server started on {self.sock.getsockname()}")
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
		"""
		Accept a connection and create a client handler for it.	
		"""
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
		"""
		The ClientHandler object handles a single client connection, and sends
		inputs to the server, and returns the server's predictions to the client.
		"""
		super().__init__()
		self.BUFFER_SIZE = config.SOCKET_BUFFER_SIZE
		self.sock = sock
		self.address = address

	def run(self):
		"""Create a new thread"""
		print(f"ClientHandler started.")
		while True:
			data = self.receive()
			if data is None or len(data) == 0:
				self.close()
				break
			data = np.array(np.frombuffer(data, dtype=bool))
			data = data.reshape(1, *config.INPUT_SHAPE)
			data = tf.convert_to_tensor(data, dtype=tf.bool)
			# make prediction
			p, v = predict(data)
			p, v = p[0].numpy().tolist(), float(v[0][0])
			response = json.dumps({"prediction": p, "value": v})
			self.send(f"{len(response):010d}".encode('ascii'))
			self.send(response.encode('ascii'))

	def receive(self):
		"""
		Receive data from the client.
		"""
		data = None
		try:
			data_length = self.sock.recv(10)
			if data_length == b'':
				# this happens if the socket connects and then closes without sending data
				return data
			data_length = int(data_length.decode("ascii"))
			data = utils.recvall(self.sock, data_length)
			if len(data) != 1216:
				data = None
				raise ValueError("Invalid data length, closing socket")
		except ConnectionResetError:
			logging.warning(f"Connection reset by peer. Client IP: {str(self.address[0])}:{str(self.address[1])}")
		except ValueError as e:
			logging.warning(e)
		return data

	def send(self, data):
		"""
		Send data to the client.
		"""
		logging.debug("Sending data...")
		self.sock.send(data)
		logging.debug("Data sent.")

	def close(self):
		"""
		Close the client connection.
		"""
		logging.info("Closing connection...")
		self.sock.close()
		logging.info("Connection closed.")
	

if __name__ == "__main__":
	# create the server socket and start the server
	s = ServerSocket(os.environ.get("SOCKET_HOST", "0.0.0.0"), int(os.environ.get("SOCKET_PORT", 5000)))
	s.start()
	