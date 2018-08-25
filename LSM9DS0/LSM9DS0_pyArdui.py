import serial
arduino = serial.Serial('/dev/tty.usbmodemFA131', 115200, timeout=.1)
#arduino = serial.Serial('/dev/tty.usbmodemFA131', 9600, timeout=.1)

def arduiData(ard = arduino):
	accelX = 0 
	accelY = 0 
	accelZ = 0
	gyroX = 0
	gyroY = 0 
	gyroZ = 0
	#pitch = 0 
	#roll = 0
	data = ard.readline()
	#print(data)
	if(data == b''):
		#print(accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll)
		return accelX, accelY, accelZ, gyroX, gyroY, gyroZ#, pitch, roll
	if(data == b'Should be 0x49D4\r\n'):
		print('connected!!')
		#print(accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll)
		return accelX, accelY, accelZ, gyroX, gyroY, gyroZ#, pitch, roll
	if(data == b"LSM9DS0 WHO_AM_I's returned: 0x49D4\r\n"):
		return accelX, accelY, accelZ, gyroX, gyroY, gyroZ#, pitch, roll
	if(len(data.split()) == 8):
		data = str(data)
		data = data[2:]
		data = data[:-5]
		data = data.split();
		if( data[0] == 'A'):
			accelX = int(data[1])
			accelY = int(data[2])		
			accelZ = int(data[3])
		if( data[4] == 'G'):
			gyroX = int(data[5])
			gyroY = int(data[6])
			gyroZ = int(data[7])
		return accelX, accelY, accelZ, gyroX, gyroY, gyroZ
	else:
		return accelX, accelY, accelZ, gyroX, gyroY, gyroZ
	'''
	try:
		#print(data)
		data = str(data)
		if(len(data) >= 15):
			data = data[2:]
			data = data[:-5]
			#print(data)
			data = data.split();
			#print(data[1])
			if( data[0] == 'A'):
				#print('wwwwwwww')
				accelX = int(data[1])
				accelY = int(data[2])
				accelZ = int(data[3])
			if( data[4] == 'G'):
				#print('ggggggggggg')
				gyroX = int(data[5])
				gyroY = int(data[6])
				gyroZ = int(data[7])
			return accelX, accelY, accelZ, gyroX, gyroY, gyroZ
	except Exception:
		pass'''
	

'''
while True:
	print(arduiData(arduino))
'''