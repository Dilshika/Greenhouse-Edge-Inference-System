export type MetricKey = "temp" | "hum" | "co2";

type PredField = "predTemp" | "predHum" | "predCo2";

export interface MetricMeta {
  label: string;
  unit: string;
  threshold: number;
  color: string;
  predField: PredField;
}