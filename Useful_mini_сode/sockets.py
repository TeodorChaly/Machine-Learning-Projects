import socket
import threading

# In programming, a socket is a way for computer programs to exchange information over the internet.
# To start this app, run sockets.py and clients.py simultaneously

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s. bind(('127.0.0.1', 55555))
s.listen()

clients = []
nicnames = []

def broadcast(message):
    for client in clients:
        client.send(message)

def handle(client):
    pass

def receive():
    while True:
        client_communication, address = s.accept()
        print(f"Connected with {str(address)}")

        client_communication.send("Nick".encode('utf-8'))
        nickname = client_communication.recv(1024)

        nicnames.append(nickname)
        clients.append(client_communication)

        print(f"Nickname of the client us {nickname}")
        broadcast(f"Hello to {nickname}!\n".encode('utf-8'))
        client_communication.send("Welcome to server".encode('utf-8'))

        thread = threading.Thread(target=handle, args=(client_communication, ))
        thread.start()