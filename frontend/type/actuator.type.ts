export type ActuatorId = "window" | "ventilation";
export type ActuatorStatus = "OPEN" | "CLOSED" | "ON" | "OFF";

export interface ActuatorState {
  status: ActuatorStatus;
  auto: boolean;
}

export type ActuatorMap = Record<ActuatorId, ActuatorState>;


