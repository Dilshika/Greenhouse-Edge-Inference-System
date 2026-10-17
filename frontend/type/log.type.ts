import {ActionSource} from './action.type';

export interface LogEntry {
  id: number;
  ts: number;
  action: string;
  reason: string;
  source: ActionSource;
}

export type NewLogEntry = Omit<LogEntry, "id" | "ts">;

