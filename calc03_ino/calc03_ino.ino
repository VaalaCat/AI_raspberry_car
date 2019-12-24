#define DIR1 8//HIGH
#define PWM1 9
#define DIR2 4//LOW
#define PWM2 5
#define DIR3 2//LOW
#define PWM3 3
#define DIR4 7//HIGH
#define PWM4 6
void setup() {
  pinMode(PWM1, OUTPUT);
  pinMode(DIR1, OUTPUT);
  pinMode(PWM2, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(PWM3, OUTPUT);
  pinMode(DIR3, OUTPUT);
  pinMode(PWM4, OUTPUT);
  pinMode(DIR4, OUTPUT);
  Serial.begin(9600);
  analogWrite(PWM1,0);
  analogWrite(PWM2,0);
  analogWrite(PWM3,0);
  analogWrite(PWM4,0);
}
char c[50];
int v[5];
const float con = 182.26;
int a[3],temp,i,j;
float omega;
void loop() {
  int cnt=-1;
  while(!Serial.available());
  if (Serial.available()) {
    while (Serial.available()) {
      c[++cnt] = Serial.read();
      delay(2);
    }
  }
  
  //for(i=0;i<=13;++i) Serial.print(c[i]);
  //Serial.println("");
  for (i = 1;i <= 2;++i) {
    temp = 0;
    for (j = 1;j <= 3;++j) {
      temp = (temp << 1) + (temp << 3) + c[(i - 1) * 4 + j] - 48;
    }
    a[i] = temp;
  }
  if (c[0] == 'L') a[1] *= -1;
  if (c[4] == 'B') a[2] *= -1;
  temp = 0;
  for (i = 1;i <= 4;++i) temp = (temp << 1) + (temp << 3) + c[8 + i] - 48;
  omega = temp;
  if (c[8] == 'R') omega *= -1;
  omega/=9999;
  v[1] = -a[1] + a[2] + omega * con;
  v[2] = a[1] + a[2] - omega * con;
  v[3] = -a[1] + a[2] - omega * con;
  v[4] = a[1] + a[2] + omega * con;
  digitalWrite(DIR1, v[1] > 0 ? HIGH : LOW);
  digitalWrite(DIR2, v[2] > 0 ? LOW : HIGH);
  digitalWrite(DIR3, v[3] > 0 ? LOW : HIGH);
  digitalWrite(DIR4, v[4] > 0 ? HIGH : LOW);
  for(i=1;i<=4;++i) if(v[i]<0) v[i]=-v[i];
  analogWrite(PWM1,v[1]);
  analogWrite(PWM2,v[2]);
  analogWrite(PWM3,v[3]);
  analogWrite(PWM4,v[4]);
  while (Serial.read() > 0)
    delay(1); /* do nothing else */
  delay(10);
}
