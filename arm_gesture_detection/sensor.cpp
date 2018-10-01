/* MPU9250 Basic Example Code
 by: Kris Winer
 date: April 1, 2014
 license: Beerware - Use this code however you'd like. If you 
 find it useful you can buy me a beer some time.
 
 Demonstrate basic MPU-9250 functionality including parameterizing the register addresses, initializing the sensor, 
 getting properly scaled accelerometer, gyroscope, and magnetometer data out. Added display functions to 
 allow display to on breadboard monitor. Addition of 9 DoF sensor fusion using open source Madgwick and 
 Mahony filter algorithms. Sketch runs on the 3.3 V 8 MHz Pro Mini and the Teensy 3.1.
 
 SDA and SCL should have external pull-up resistors (to 3.3V).
 10k resistors are on the EMSENSR-9250 breakout board.
 
 Hardware setup:
 MPU9250 Breakout --------- Arduino
 VDD ---------------------- 3.3V
 VDDI --------------------- 3.3V
 SDA ----------------------- A4
 SCL ----------------------- A5
 GND ---------------------- GND
 
 Note: The MPU9250 is an I2C sensor and uses the Arduino Wire library. 
 Because the sensor is not 5V tolerant, we are using a 3.3 V 8 MHz Pro Mini or a 3.3 V Teensy 3.1.
 We have disabled the internal pull-ups used by the Wire library in the Wire.h/twi.c utility file.
 We are also using the 400 kHz fast I2C mode by setting the TWI_FREQ  to 400000L /twi.h utility file.
 */
 
#include "sensor.h"

Sensor::Sensor(){
  float sum = 0;
  uint32_t sumCount = 0;
  
  MPU9250 mpu9250;
  Timer t;
  Serial pc(USBTX, USBRX); // tx, rx
        
  pc.baud(38400); 
  //Set up I2C
  i2c.frequency(400000);  // use fast (400 kHz) I2C  
  
  t.start();        
  
  uint8_t whoami = mpu9250.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);  // Read WHO_AM_I register for MPU-9250
   // pc.printf("I AM 0x%x\n\r", whoami); pc.printf("I SHOULD BE 0x73\n\r");
  
  if (whoami == 0x73) {// WHO_AM_I should always be 0x73
    wait(1);
    mpu9250.resetMPU9250(); // Reset registers to default in preparation for device calibration
    mpu9250.MPU9250SelfTest(SelfTest); // Start by performing self test and reporting values
    mpu9250.calibrateMPU9250(gyroBias, accelBias); // Calibrate gyro and accelerometers, load biases in bias registers  
    wait(2);
    mpu9250.initMPU9250(); 
    mpu9250.initAK8963(magCalibration);
    wait(1);
  }
  else {
    pc.printf("Could not connect to MPU9250: \n\r");
    pc.printf("%#x \n",  whoami);
    while(1) ; // Loop forever if communication doesn't happen
  }

  mpu9250.getAres(); // Get accelerometer sensitivity
  mpu9250.getGres(); // Get gyro sensitivity
  mpu9250.getMres(); // Get magnetometer sensitivity
  magbias[0] = +470.;  // User environmental x-axis correction in milliGauss, should be automatically calculated
  magbias[1] = +120.;  // User environmental x-axis correction in milliGauss
  magbias[2] = +125.;  // User environmental x-axis correction in milliGauss
}
    
float* Sensor::read(){
  // If intPin goes high, all data registers have new data
  if(mpu9250.readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01) {  // On interrupt, check if data ready interrupt
      
    mpu9250.readAccelData(accelCount);  // Read the x/y/z adc values   
    // Now we'll calculate the accleration value into actual g's
    ax = (float)accelCount[0]*aRes - accelBias[0];  // get actual g value, this depends on scale being set
    ay = (float)accelCount[1]*aRes - accelBias[1];   
    az = (float)accelCount[2]*aRes - accelBias[2];  
       
    mpu9250.readGyroData(gyroCount);  // Read the x/y/z adc values
    // Calculate the gyro value into actual degrees per second
    gx = (float)gyroCount[0]*gRes - gyroBias[0];  // get actual gyro value, this depends on scale being set
    gy = (float)gyroCount[1]*gRes - gyroBias[1];  
    gz = (float)gyroCount[2]*gRes - gyroBias[2];   

    mpu9250.readMagData(magCount);  // Read the x/y/z adc values   
    // Calculate the magnetometer values in milliGauss
    // Include factory calibration per data sheet and user environmental corrections
    mx = (float)magCount[0]*mRes*magCalibration[0] - magbias[0];  // get actual magnetometer value, this depends on scale being set
    my = (float)magCount[1]*mRes*magCalibration[1] - magbias[1];  
    mz = (float)magCount[2]*mRes*magCalibration[2] - magbias[2];   
  }
 
  Now = t.read_us();
  deltat = (float)((Now - lastUpdate)/1000000.0f) ; // set integration time by time elapsed since last filter update
  lastUpdate = Now;
  
  sum += deltat;
  sumCount++;
    
  // Pass gyro rate as rad/s
  //  mpu9250.MadgwickQuaternionUpdate(ax, ay, az, gx*PI/180.0f, gy*PI/180.0f, gz*PI/180.0f,  my,  mx, mz);
  mpu9250.MahonyQuaternionUpdate(ax, ay, az, gx*PI/180.0f, gy*PI/180.0f, gz*PI/180.0f, my, mx, mz);

  // Serial print and/or display at 0.5 s rate independent of data rates
  delt_t = t.read_ms() - count;

  float* data = new float[16];

    
  data[0] = ax;
  data[1] = ay;
  data[2] = az;
  data[3] = gx;
  data[4] = gy;
  data[5] = gz;
  data[6] = mx;
  data[7] = my;
  data[8] = mz;

       
  tempCount = mpu9250.readTempData();  // Read the adc values
  temperature = ((float) tempCount) / 333.87f + 21.0f; // Temperature in degrees Centigrade
  // pc.printf(" temperature = %f  C\n\r", temperature); 
  data[9] = q[0];
  data[10] = q[1];
  data[11] = q[2];
  data[12] = q[3];
  
  // Define output variables from updated quaternion---these are Tait-Bryan angles, commonly used in aircraft orientation.
  // In this coordinate system, the positive z-axis is down toward Earth. 
  // Yaw is the angle between Sensor x-axis and Earth magnetic North (or true North if corrected for local declination, looking down on the sensor positive yaw is counterclockwise.
  // Pitch is angle between sensor x-axis and Earth ground plane, toward the Earth is positive, up toward the sky is negative.
  // Roll is angle between sensor y-axis and Earth ground plane, y-axis up is positive roll.
  // These arise from the definition of the homogeneous rotation matrix constructed from quaternions.
  // Tait-Bryan angles as well as Euler angles are non-commutative; that is, the get the correct orientation the rotations must be
  // applied in the correct order which for this configuration is yaw, pitch, and then roll.
  // For more see http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles which has additional links.
  yaw   = atan2(2.0f * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);   
  pitch = -asin(2.0f * (q[1] * q[3] - q[0] * q[2]));
  roll  = atan2(2.0f * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
  pitch *= 180.0f / PI;
  yaw   *= 180.0f / PI; 
  yaw   -= 13.8f; // Declination at Danville, California is 13 degrees 48 minutes and 47 seconds on 2014-04-04
  roll  *= 180.0f / PI;
       
  data[9] = q[0];
  data[10] = q[1];
  data[11] = q[2];
  data[12] = q[3];
  // pc.printf("average rate = %f\n\r", (float) sumCount/sum);
  // pc.printf("===========================================\n\r");

  myled= !myled;
  count = t.read_ms(); 
  if(count > 1<<21) {
    t.start(); // start the timer over again if ~30 minutes has passed
    count = 0;
    deltat= 0;
    lastUpdate = t.read_us();
  }
  sum = 0;
  sumCount = 0; 
}
 
