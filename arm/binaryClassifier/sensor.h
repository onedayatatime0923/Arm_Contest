#include <vector>
#include "mbed.h"
#include "MPU9250.h"


float sum = 0;
int sumCount = 0;
char buffer[14];
MPU9250 mpu9250;
Timer t;

void connect_MPU9250(Serial pc) {
   i2c.frequency(400000);  // use fast (400 kHz) I2C  
   t.start();        
   uint8_t whoami = mpu9250.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);  // Read WHO_AM_I register for MPU-9250
   // pc.printf("I AM 0x%x\n\r", whoami); pc.printf("I SHOULD BE 0x73\n\r");
   if (whoami == 0x73) {// WHO_AM_I should always be 0x73
      // pc.printf("MPU9250 WHO_AM_I is 0x%x\n\r", whoami);
      // sprintf(buffer, "0x%x", whoami);
      wait(1);
      mpu9250.resetMPU9250(); // Reset registers to default in preparation for device calibration
      mpu9250.calibrateMPU9250(gyroBias, accelBias); // Calibrate gyro and accelerometers, load biases in bias registers  
      wait(2);
      mpu9250.initMPU9250(); 
      mpu9250.initAK8963(magCalibration);
      wait(1);
   }
   else {
      pc.printf("Could not connect to MPU9250: \n\r");
      pc.printf("%#x \n",  whoami);
      sprintf(buffer, "WHO_AM_I 0x%x", whoami);
      while(1) ; // Loop forever if communication doesn't happen
   }
   mpu9250.getAres(); // Get accelerometer sensitivity
   mpu9250.getGres(); // Get gyro sensitivity
   mpu9250.getMres(); // Get magnetometer sensitivity
   magbias[0] = +470.;  // User environmental x-axis correction in milliGauss, should be automatically calculated
   magbias[1] = +120.;  // User environmental x-axis correction in milliGauss
   magbias[2] = +125.;  // User environmental x-axis correction in milliGauss
}

vector<float> read_data(Serial& pc) {
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
     // // Include factory calibration per data sheet and user environmental corrections
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
           
  tempCount = mpu9250.readTempData();  // Read the adc values
  temperature = ((float) tempCount) / 333.87f + 21.0f; // Temperature in degrees Centigrade

  yaw   = atan2(2.0f * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);   
  pitch = -asin(2.0f * (q[1] * q[3] - q[0] * q[2]));
  roll  = atan2(2.0f * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
  pitch *= 180.0f / PI;
  yaw   *= 180.0f / PI; 
  yaw   -= 13.8f; // Declination at Danville, California is 13 degrees 48 minutes and 47 seconds on 2014-04-04
  roll  *= 180.0f / PI;
  /* 
  pc.printf("YPR: ");
  pc.printf("%f ", yaw);
  pc.printf("%f ", pitch);
  pc.printf("%f\n", roll);
  */
  vector<float> data;
  data.reserve()
  data.push_back(ax);
  data.push_back(ay);
  data.push_back(az);
  data.push_back(gx);
  data.push_back(gy);
  data.push_back(gz);
  data.push_back(mx);
  data.push_back(my);
  data.push_back(mz);
  data.push_back(q[0]);
  data.push_back(q[1]);
  data.push_back(q[2]);
  data.push_back(q[3]);
  data.push_back(yaw);
  data.push_back(pitch);
  data.push_back(roll);

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
  return data
}
