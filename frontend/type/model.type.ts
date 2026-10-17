import { LogEntry } from "./log.type";

export type ModelName = "XGBoost" | "GRU" | "LSTM";

export interface EdgeHealth {
  model: ModelName;
  accuracy: number;
  cpu: number;
  memory: number;
  latency: number;
  heat: number;
}