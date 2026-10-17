import { ActuatorId, ActuatorStatus } from "./actuator.type";

export interface PendingAction {
  id: number;
  action: string;
  target: ActuatorId;
  newStatus: ActuatorStatus;
  reason: string;
  deadline: number;
}

export type ActionSource = "AI Autonomous" | "Telegram Bot" | "React Manual Override";
