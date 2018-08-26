import socket
import pyautogui
import time
pyautogui.moveTo(100, 150)
HOST, PORT = "127.0.0.1", 11111
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PORT))

datalost=0
dataprev=0
shift=False

while(True):
	time.sleep(1)
	data, addr = s.recvfrom(2048)
	data=data.decode("utf-8")
	data=int(data)
	print(data)
	if data==0:
		print(0)
		pyautogui.press('0')
	elif data==1:
		print(1)
		pyautogui.press('1')
	elif data==2:
		print(2)
		pyautogui.press('2')
	elif data==3:
		print(3)
		pyautogui.press('3')
	elif data==4:
		print(4)
		pyautogui.press('4')
	elif data==5:
		print(5)
		pyautogui.press('5')
	elif data==6:
		print(6)
		pyautogui.press('6')
	elif data==7:
		print(7)
		pyautogui.press('7')
	elif data==8:
		print(8)
		pyautogui.press('8')
	elif data==9:
		print(9)
		pyautogui.press('9')
	elif data==10:
		print(10)
		pyautogui.press('left')
	elif data==11:
		print(11)
		pyautogui.press('right')
	elif data==12:
		print(12)
		pyautogui.press('backspace')
	elif data==13:
		print('space')
		pyautogui.press('space')
	elif data==14:
		print('...')
		pyautogui.press(['.', '.', '.'])
	elif data==15:
		print('gg')
		pyautogui.hotkey('command', 'space')
		pyautogui.hotkey('shift', '.')
		pyautogui.hotkey('command', 'space')
	elif data==16:
		print('b')
		pyautogui.press('1')
	elif data==17:
		print('p')
		pyautogui.press('q')
	elif data==18:
		print('m')
		pyautogui.press('a')
	elif data==19:
		print('ao')
		pyautogui.press('l')
	elif data==20:
		print('a')
		pyautogui.press('8')

	time.sleep(0.5)


s.close()  