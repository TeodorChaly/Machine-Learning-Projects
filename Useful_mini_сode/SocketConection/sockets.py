import socket
# In programming, a socket is a way for computer programs to exchange information over the internet.
# To start this app, run sockets.py and clients.py simultaneously

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s. bind(('127.0.0.1', 55555))
s.listen()

while True:
    client, address = s.accept()
    print(f"Connected to {address}")
    client.send("You are connected!" .encode())
    client.close()