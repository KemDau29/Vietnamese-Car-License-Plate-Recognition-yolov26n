#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); 
Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
  myServo.write(0); // Đóng ban đầu
  
  lcd.init();
  lcd.backlight();
  lcd.print("SYSTEM READY");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // Đọc cả dòng
    
    if (data.startsWith("O:")) {
      String plate = data.substring(2); // Cắt lấy phần biển số
      myServo.write(90); // Mở barrier
      
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("PLATE: " + plate);
      lcd.setCursor(0, 1);
      lcd.print("BARRIER: OPENED");
    } 
    else if (data == "C") {
      myServo.write(0); // Đóng barrier
      
      lcd.clear();
      lcd.print("BARRIER: CLOSED");
      delay(1000);
      lcd.clear();
      lcd.print("SYSTEM READY");
    }
  }
}