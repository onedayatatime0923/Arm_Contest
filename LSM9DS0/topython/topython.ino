int analogPin = 3;     // potentiometer wiper (middle terminal) connected to analog pin 3
                       // outside leads to ground and +5V
int val = 0;

void setup() {
  Serial.begin(115200);
}
void loop() {
  val = analogRead(analogPin);
  /*
  if(Serial.available() > 0) {
    char data = Serial.read();
    char str[2];
    str[0] = data;
    str[1] = '\0';
   */
   Serial.println(val);
   delay(1);
   //}
}


