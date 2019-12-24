import socket

address = ('127.0.0.1', 5555)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(address)
s.listen(5)

ss, addr = s.accept()
print(addr)

ss.send(bytes('1234',encoding='ascii'))
ra = ss.recv(512)
print(ra)


ss.close()
s.close()
