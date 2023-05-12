#include <ESP8266WiFi.h>

#ifndef STASSID
#define STASSID "soccer"
#define STAPSK "12345678"
#define NAME "B"
#endif

const char* host = "192.168.0.2";
const uint16_t port = 2023;
// buffers for receiving and sending data
int act[] = {0, 0};
float percent;
unsigned long timer = 0;
bool sended = false;
WiFiClient client;

void setup() {
  //Connect to AP
  analogWriteFreq(1000);
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(STASSID, STAPSK);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }

  //pinMode set
  pinMode(D2, OUTPUT);
  pinMode(D3, OUTPUT);
  pinMode(D4, OUTPUT);
  pinMode(D5, OUTPUT);
  pinMode(D6, OUTPUT);
  pinMode(D7, OUTPUT);

  //Connected
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}
//server에 연결하고 명령을 기다리다가 Act가 들어오면 Act, Check가 들어오면 Reply
void loop() {
  // Read Battery Voltage
  float volt = analogRead(A0) / 1024.0 * 3.3 * 3;
  percent = (volt - 6) / 2.4 ; //6v to 8.4v

  //server connected
  if (client.connected())
  {    
    //HeartBeat
    client.write("");
    delay(50);
    if (millis() > timer + 5000)
    {
      // Checking Packet not recevied in 5 seconds
      Serial.println("Connection Timeout");
      client.stop();
      return;
    }
    while (client.connected() && client.available()) {
      timer = millis();
      char c = client.read();
      Serial.print("Received : ");
      Serial.println(c);
      //act
      if (c == 'A')
      {
        char string[32];
        int avail = client.available();
        for (int i=0; i<avail; i++)
        {
          string[i] = client.read();
        }
        string[avail] = '\0';
        String s = String(string);
        int idx = s.indexOf('&');
        int R = s.substring(idx+1).toInt();
        int L = s.substring(0, idx).toInt();
        String ss ="ACT: ";
        ss += String(L);
        ss += " ";
        ss += String(R);
        Serial.println(ss);
        Move(L, R);
      }
      else if (c == 'C')
      {
          String send = "";
          send += NAME;
          send += "-";
          send += String(percent, 2);
          client.print(send);
      }
      else if (c == 'H')
      {
          Serial.println("Server says hello");
          client.write(NAME);
      }
    }  
  }
  //server not connected
  else
  {
    Serial.print("Connect to TCP Server");
    client.stop();
    while (!client.connect(host, port)) {
      Serial.print(".");
      delay(500);
    }
    Serial.println("\nConnected to Server");
    timer = millis();
  }
}

void Move(int L, int R)
{
  analogWrite(D7, int(abs(R)));
  analogWrite(D4, int(abs(L)));
  if (L < 0)
  {
    digitalWrite(D2, HIGH);
    digitalWrite(D3, LOW);
  }
  else if (L > 0)
  {
    digitalWrite(D2, LOW);
    digitalWrite(D3, HIGH);
  }
  else
  {
    digitalWrite(D2, HIGH);
    digitalWrite(D3, HIGH);
  }

  if (R < 0)
  {
    digitalWrite(D5, HIGH);
    digitalWrite(D6, LOW);
  }
  else if (R > 0)
  {
    digitalWrite(D5, LOW);
    digitalWrite(D6, HIGH);
  }
  else
  {
    digitalWrite(D5, HIGH);
    digitalWrite(D6, HIGH);
  }
}