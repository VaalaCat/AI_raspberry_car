import socket

address = ('127.0.0.1', 5555)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(address)

data = s.recv(512)

print(data)

s.send(bytes('haha',encoding='ascii'))

s.close()
