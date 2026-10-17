export interface RawReading {
  t: number;
  temp: number;
  hum: number;
  co2: number;
}

export interface Reading extends RawReading {
  predTemp: number;
  predHum: number;
  predCo2: number;
}
