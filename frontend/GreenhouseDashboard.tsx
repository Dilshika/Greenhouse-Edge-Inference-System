import { useState, useEffect, useRef } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import {
  Thermometer,
  Droplets,
  Wind,
  AlertTriangle,
  Cpu,
  Gauge,
  MemoryStick,
  Flame,
  Blinds,
  Check,
  X,
  Radio,
  type LucideIcon,
} from "lucide-react";
import { ActuatorMap, ActuatorId, ActuatorStatus, ActuatorState } from "./type/actuator.type";
import { MetricKey, MetricMeta } from "./type/metric.type";
import { RawReading, Reading } from "./type/reading.type";
import {ActionSource, PendingAction} from './type/action.type';
import { LogEntry, NewLogEntry } from "./type/log.type";
import {EdgeHealth} from './type/model.type';


// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const TICK_MS = 2200; // compressed stand-in for a 5-minute sensor read, for live demo purposes
const HISTORY_LEN = 60;
const APPROVAL_WINDOW_S = 12;

const METRICS: Record<MetricKey, MetricMeta> = {
  temp: { label: "Temperature", unit: "°C", threshold: 30, color: "#e3a857", predField: "predTemp" },
  hum: { label: "Humidity", unit: "%", threshold: 85, color: "#5fa0b8", predField: "predHum" },
  co2: { label: "CO2", unit: "ppm", threshold: 900, color: "#63b087", predField: "predCo2" },
};

const METRIC_KEYS = Object.keys(METRICS) as MetricKey[];

const TELEGRAM_REASONS: string[] = [
  "Grower confirmed vent close via Telegram",
  "Grower requested ventilation boost via Telegram",
];

// ---------------------------------------------------------------------------
// Helpers + simulated telemetry (stand-in for the wss feed from the Pi simulator)
// ---------------------------------------------------------------------------

function r1(n: number): number {
  return Math.round(n * 10) / 10;
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

function forecast(rows: RawReading[], field: MetricKey): number {
  const n = rows.length;
  const last = rows[n - 1][field];
  const prior = rows[Math.max(0, n - 5)][field];
  const slope = (last - prior) / 4;
  const noise = (Math.random() - 0.5) * (field === "co2" ? 6 : 0.4);
  return last + slope * 3 + noise;
}

function withForecast(rows: RawReading[]): Reading {
  const base = rows[rows.length - 1];
  return {
    ...base,
    predTemp: r1(forecast(rows, "temp")),
    predHum: r1(forecast(rows, "hum")),
    predCo2: Math.round(forecast(rows, "co2")),
  };
}

function seedHistory(): Reading[] {
  const now = Date.now();
  const raw: RawReading[] = [];
  let temp = 24.5;
  let hum = 66;
  let co2 = 640;
  for (let i = HISTORY_LEN; i > 0; i--) {
    temp += (Math.random() - 0.5) * 0.35;
    hum += (Math.random() - 0.5) * 1.0;
    co2 += (Math.random() - 0.5) * 10;
    raw.push({ t: now - i * TICK_MS, temp: r1(temp), hum: r1(hum), co2: Math.round(co2) });
  }
  return raw.map((_, idx) => withForecast(raw.slice(0, idx + 1)));
}

function sourceClass(source: ActionSource): string {
  if (source === "AI Autonomous") return "ai";
  if (source === "Telegram Bot") return "telegram";
  return "manual";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GreenhouseDashboard() {
  const [history, setHistory] = useState<Reading[]>(seedHistory);
  const [connected, setConnected] = useState<boolean>(true);
  const [metricTab, setMetricTab] = useState<MetricKey>("temp");
  const [actuators, setActuators] = useState<ActuatorMap>({
    window: { status: "CLOSED", auto: true },
    ventilation: { status: "OFF", auto: true },
  });
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [now, setNow] = useState<number>(Date.now());
  const [log, setLog] = useState<LogEntry[]>([
    {
      id: 1,
      ts: Date.now() - 90000,
      action: "System initialized",
      reason: "Edge service connected to broker",
      source: "AI Autonomous",
    },
  ]);
  const [health, setHealth] = useState<EdgeHealth>({
    model: "GRU",
    accuracy: 94.2,
    cpu: 31,
    memory: 44,
    latency: 46,
    heat: 51,
  });
  const logId = useRef<number>(2);
  const pendingId = useRef<number>(1);
  const lastTrigger = useRef<number>(0);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  // telemetry tick
  useEffect(() => {
    const timer = setInterval(() => {
      setHistory((prev) => {
        const last = prev[prev.length - 1];
        const next: RawReading = {
          t: Date.now(),
          temp: r1(clamp(last.temp + (Math.random() - 0.46) * 0.45, 18, 33)),
          hum: r1(clamp(last.hum + (Math.random() - 0.5) * 1.5, 40, 92)),
          co2: Math.round(clamp(last.co2 + (Math.random() - 0.5) * 16, 450, 980)),
        };
        return [...prev.slice(1), withForecast([...prev.slice(1), next])];
      });

      setHealth((h) => ({
        model: h.model,
        accuracy: clamp(h.accuracy + (Math.random() - 0.5) * 0.6, 88, 97),
        cpu: Math.round(clamp(h.cpu + (Math.random() - 0.5) * 8, 15, 78)),
        memory: Math.round(clamp(h.memory + (Math.random() - 0.5) * 4, 30, 70)),
        latency: Math.round(clamp(h.latency + (Math.random() - 0.5) * 10, 20, 120)),
        heat: Math.round(clamp(h.heat + (Math.random() - 0.5) * 3, 40, 68)),
      }));

      if (Math.random() < 0.04) {
        setConnected(false);
        setTimeout(() => setConnected(true), 1500);
      }

      if (Math.random() < 0.03) {
        pushLog({
          action: Math.random() < 0.5 ? "Ventilation adjusted" : "Window state confirmed",
          reason: TELEGRAM_REASONS[Math.floor(Math.random() * TELEGRAM_REASONS.length)],
          source: "Telegram Bot",
        });
      }
    }, TICK_MS);
    return () => clearInterval(timer);
  }, []);

  // anomaly detection -> raise a pending approval
  useEffect(() => {
    if (pending) return;
    const latest = history[history.length - 1];
    const cooldownOk = Date.now() - lastTrigger.current > 20000;
    if (!cooldownOk) return;
    if (latest.predTemp > METRICS.temp.threshold && actuators.window.auto) {
      lastTrigger.current = Date.now();
      setPending({
        id: pendingId.current++,
        action: "Open roof vent",
        target: "window",
        newStatus: "OPEN",
        reason: `Predicted temperature ${latest.predTemp.toFixed(1)}°C exceeds ${METRICS.temp.threshold}°C threshold`,
        deadline: Date.now() + APPROVAL_WINDOW_S * 1000,
      });
    } else if (latest.predCo2 > METRICS.co2.threshold && actuators.ventilation.auto) {
      lastTrigger.current = Date.now();
      setPending({
        id: pendingId.current++,
        action: "Increase ventilation",
        target: "ventilation",
        newStatus: "ON",
        reason: `Predicted CO2 ${Math.round(latest.predCo2)}ppm exceeds ${METRICS.co2.threshold}ppm threshold`,
        deadline: Date.now() + APPROVAL_WINDOW_S * 1000,
      });
    }
  }, [history, pending, actuators]);

  // auto-execute the pending action if the approval window elapses
  useEffect(() => {
    if (!pending) return;
    if (now >= pending.deadline) {
      applyActuator(pending.target, pending.newStatus);
      pushLog({
        action: pending.action,
        reason: `${pending.reason} — auto-executed after no response`,
        source: "AI Autonomous",
      });
      setPending(null);
    }
  }, [now, pending]);

  function pushLog(entry: NewLogEntry): void {
    setLog((prev) => [{ id: logId.current++, ts: Date.now(), ...entry }, ...prev.slice(0, 14)]);
  }

  function applyActuator(id: ActuatorId, status: ActuatorStatus): void {
    setActuators((prev) => ({ ...prev, [id]: { ...prev[id], status } }));
  }

  function approvePending(): void {
    if (!pending) return;
    applyActuator(pending.target, pending.newStatus);
    pushLog({
      action: pending.action,
      reason: `${pending.reason} — approved via dashboard`,
      source: "AI Autonomous",
    });
    setPending(null);
  }

  function rejectPending(): void {
    if (!pending) return;
    pushLog({
      action: `Rejected: ${pending.action}`,
      reason: pending.reason,
      source: "React Manual Override",
    });
    setPending(null);
  }

  function toggleAuto(id: ActuatorId): void {
    setActuators((prev) => ({ ...prev, [id]: { ...prev[id], auto: !prev[id].auto } }));
  }

  function manualSet(id: ActuatorId, status: ActuatorStatus): void {
    applyActuator(id, status);
    pushLog({
      action: `${id === "window" ? "Window" : "Ventilation"} set to ${status}`,
      reason: "Operator command from dashboard",
      source: "React Manual Override",
    });
  }

  const latest = history[history.length - 1];
  const activeMetric = METRICS[metricTab];

  return (
    <div className="gh-root">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        .gh-root {
          --void:#0c1210; --panel:#121a16; --panel-raised:#182019; --line:#263229;
          --text:#e9ede7; --text-muted:#859487; --sage:#63b087; --amber:#e3a857;
          --blue:#5fa0b8; --red:#cc5b48;
          font-family:'Space Grotesk',sans-serif; background:var(--void);
          background-image: radial-gradient(circle at 1px 1px, rgba(233,237,231,0.035) 1px, transparent 0);
          background-size:22px 22px; color:var(--text); min-height:100vh; padding:20px; box-sizing:border-box;
        }
        .gh-root * { box-sizing:border-box; }
        .mono { font-family:'IBM Plex Mono',monospace; }
        .gh-header { display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:12px; border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:18px; }
        .gh-title { font-size:21px; font-weight:600; letter-spacing:-0.01em; }
        .gh-subtitle { color:var(--text-muted); font-size:13px; margin-top:3px; }
        .gh-header-right { display:flex; align-items:center; gap:16px; }
        .status-pill { display:flex; align-items:center; gap:7px; font-size:12px; color:var(--text-muted); padding:5px 10px; border:1px solid var(--line); border-radius:3px; }
        .dot { width:7px; height:7px; border-radius:50%; }
        .dot.ok { background:var(--sage); box-shadow:0 0 6px var(--sage); }
        .dot.down { background:var(--red); box-shadow:0 0 6px var(--red); }
        .panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:14px 16px; }
        .panel-title { font-size:12.5px; color:var(--text-muted); font-weight:500; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; }
        .grid { display:grid; gap:14px; margin-bottom:14px; }
        .row1 { grid-template-columns:repeat(3,1fr); }
        .row3 { grid-template-columns:1fr 1fr; align-items:start; }

        .forecast-card { display:flex; flex-direction:column; gap:10px; }
        .fc-label { font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px; }
        .fc-flow { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
        .fc-current { font-size:24px; font-weight:600; }
        .fc-arrow { color:var(--text-muted); font-size:16px; }
        .fc-predicted { font-size:20px; font-weight:600; }
        .fc-predicted.danger { color:var(--amber); }
        .fc-unit { font-size:12px; color:var(--text-muted); font-weight:400; }
        .fc-warn { color:var(--amber); }
        .fc-sub { font-size:10.5px; color:var(--text-muted); }

        .tab-row { display:flex; gap:6px; }
        .tab-btn { font-family:'Space Grotesk',sans-serif; font-size:11.5px; padding:5px 10px; border-radius:3px; border:1px solid var(--line); background:transparent; color:var(--text-muted); cursor:pointer; }
        .tab-btn.active { background:var(--panel-raised); color:var(--text); }
        .legend-row { display:flex; gap:14px; font-size:11px; color:var(--text-muted); margin-top:8px; }
        .legend-swatch { width:9px; height:2.5px; display:inline-block; margin-right:5px; vertical-align:middle; }

        .actuator-row { display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--line); }
        .actuator-left { display:flex; align-items:center; gap:10px; }
        .actuator-name { font-size:13.5px; }
        .status-tag { font-size:10.5px; padding:2px 7px; border-radius:2px; border:1px solid var(--line); color:var(--text-muted); }
        .status-tag.live { color:var(--sage); border-color:var(--sage); }
        .actuator-controls { display:flex; align-items:center; gap:8px; }
        .mode-toggle { display:flex; border:1px solid var(--line); border-radius:3px; overflow:hidden; }
        .mode-btn { font-size:10.5px; padding:4px 8px; color:var(--text-muted); background:transparent; border:none; cursor:pointer; font-family:'Space Grotesk',sans-serif; }
        .mode-btn.active { background:var(--panel-raised); color:var(--text); }
        .manual-btns { display:flex; gap:4px; }
        .manual-btn { font-size:10.5px; padding:4px 8px; border:1px solid var(--line); background:var(--panel-raised); color:var(--text-muted); border-radius:3px; cursor:pointer; }
        .manual-btn:disabled { opacity:0.35; cursor:not-allowed; }
        .manual-btn.on { color:var(--text); border-color:var(--sage); }

        .approval-box { border:1px solid var(--amber); background:rgba(227,168,87,0.08); border-radius:4px; padding:12px 14px; margin-bottom:14px; }
        .approval-head { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:500; color:var(--amber); }
        .approval-reason { font-size:12px; color:var(--text-muted); margin:6px 0 10px; }
        .approval-timer-track { height:4px; background:var(--line); border-radius:2px; overflow:hidden; margin-bottom:10px; }
        .approval-timer-fill { height:100%; background:var(--amber); transition:width 1s linear; }
        .approval-actions { display:flex; gap:8px; }
        .approve-btn, .reject-btn { flex:1; display:flex; align-items:center; justify-content:center; gap:6px; padding:7px; border-radius:3px; font-size:12.5px; cursor:pointer; border:1px solid var(--line); font-family:'Space Grotesk',sans-serif; }
        .approve-btn { background:rgba(99,176,135,0.15); border-color:var(--sage); color:var(--sage); }
        .reject-btn { background:rgba(204,91,72,0.12); border-color:var(--red); color:var(--red); }

        table.audit { width:100%; border-collapse:collapse; font-size:11.5px; }
        table.audit th { text-align:left; color:var(--text-muted); font-weight:500; padding:5px 6px; border-bottom:1px solid var(--line); }
        table.audit td { padding:6px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
        table.audit tr:last-child td { border-bottom:none; }
        .src-tag { font-size:10px; padding:2px 6px; border-radius:2px; white-space:nowrap; }
        .src-tag.ai { color:var(--sage); border:1px solid var(--sage); }
        .src-tag.telegram { color:var(--blue); border:1px solid var(--blue); }
        .src-tag.manual { color:var(--amber); border:1px solid var(--amber); }
        .audit-scroll { max-height:190px; overflow-y:auto; }

        .health-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
        .health-stat { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-muted); }
        .health-stat .mono { color:var(--text); font-size:13px; }
        .model-badge { display:inline-flex; align-items:center; gap:6px; font-size:12.5px; background:var(--panel-raised); border:1px solid var(--line); padding:4px 10px; border-radius:3px; }

        @media (max-width:980px) { .row1 { grid-template-columns:1fr; } .row3 { grid-template-columns:1fr; } }
      `}</style>

      <div className="gh-header">
        <div>
          <div className="gh-title">Greenhouse edge inference console</div>
          <div className="gh-subtitle">Live wss feed · greenhouse-edge-01 · +15 min forecast horizon</div>
        </div>
        <div className="gh-header-right">
          <div className="status-pill">
            <Radio size={12} /> Backend API + MongoDB
          </div>
          <div className="status-pill">
            <span className={`dot ${connected ? "ok" : "down"}`} />
            {connected ? "wss connected" : "Reconnecting…"}
          </div>
          <div className="mono" style={{ fontSize: 13, color: "var(--text-muted)" }}>
            {new Date(now).toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Panel 1: live sensor + 15m forecast cards */}
      <div className="grid row1">
        <ForecastCard icon={Thermometer} metricKey="temp" current={latest.temp} predicted={latest.predTemp} />
        <ForecastCard icon={Droplets} metricKey="hum" current={latest.hum} predicted={latest.predHum} />
        <ForecastCard icon={Wind} metricKey="co2" current={latest.co2} predicted={latest.predCo2} />
      </div>

      {/* Panel 2: historical vs predicted line chart */}
      <div className="panel" style={{ marginBottom: 14 }}>
        <div className="panel-title">
          <span>Historical vs. predicted: {activeMetric.label}</span>
          <div className="tab-row">
            {METRIC_KEYS.map((key) => (
              <button
                key={key}
                className={`tab-btn ${metricTab === key ? "active" : ""}`}
                onClick={() => setMetricTab(key)}
              >
                {METRICS[key].label}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={history} margin={{ top: 4, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tickFormatter={(v: number) =>
                new Date(v).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              }
              stroke="var(--text-muted)"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              minTickGap={50}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              width={38}
            />
            <Tooltip
              contentStyle={{
                background: "var(--panel-raised)",
                border: "1px solid var(--line)",
                borderRadius: 4,
                fontSize: 12,
              }}
              labelFormatter={(v: number) => new Date(v).toLocaleTimeString()}
            />
            <ReferenceLine y={activeMetric.threshold} stroke="var(--red)" strokeDasharray="5 4" strokeWidth={1.5} />
            <Line
              type="monotone"
              dataKey={metricTab}
              name="Raw sensor reading"
              stroke="#5fa0b8"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey={activeMetric.predField}
              name="AI prediction"
              stroke="#e3a857"
              dot={false}
              strokeWidth={2}
              strokeDasharray="4 3"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="legend-row">
          <span>
            <span className="legend-swatch" style={{ background: "#5fa0b8" }} />
            Raw sensor reading
          </span>
          <span>
            <span className="legend-swatch" style={{ background: "#e3a857" }} />
            AI predicted curve
          </span>
          <span>
            <span className="legend-swatch" style={{ background: "var(--red)" }} />
            Danger threshold ({activeMetric.threshold}
            {activeMetric.unit})
          </span>
        </div>
      </div>

      <div className="grid row3">
        {/* Panel 3: hardware control and manual override */}
        <div className="panel">
          <div className="panel-title">Hardware control &amp; manual override</div>

          {pending && (
            <div className="approval-box">
              <div className="approval-head">
                <AlertTriangle size={15} /> AI action pending approval
              </div>
              <div className="approval-reason">
                {pending.action}: {pending.reason}
              </div>
              <div className="approval-timer-track">
                <div
                  className="approval-timer-fill"
                  style={{
                    width: `${Math.max(0, (pending.deadline - now) / (APPROVAL_WINDOW_S * 1000)) * 100}%`,
                  }}
                />
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
                Auto-executes in {Math.max(0, Math.ceil((pending.deadline - now) / 1000))}s if no response
              </div>
              <div className="approval-actions">
                <div className="approve-btn" onClick={approvePending}>
                  <Check size={13} /> Approve
                </div>
                <div className="reject-btn" onClick={rejectPending}>
                  <X size={13} /> Reject
                </div>
              </div>
            </div>
          )}

          <ActuatorRow
            icon={Blinds}
            label="Window"
            id="window"
            state={actuators.window}
            onLabel="OPEN"
            offLabel="CLOSED"
            toggleAuto={toggleAuto}
            manualSet={manualSet}
          />
          <ActuatorRow
            icon={Flame}
            label="Ventilation"
            id="ventilation"
            state={actuators.ventilation}
            onLabel="ON"
            offLabel="OFF"
            toggleAuto={toggleAuto}
            manualSet={manualSet}
          />
        </div>

        {/* Panel 4: decision audit log and edge health */}
        <div className="panel">
          <div className="panel-title">Decision audit log &amp; edge health</div>
          <div className="audit-scroll">
            <table className="audit">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Action</th>
                  <th>Reason</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {log.map((entry) => (
                  <tr key={entry.id}>
                    <td className="mono">{new Date(entry.ts).toLocaleTimeString()}</td>
                    <td>{entry.action}</td>
                    <td style={{ color: "var(--text-muted)" }}>{entry.reason}</td>
                    <td>
                      <span className={`src-tag ${sourceClass(entry.source)}`}>{entry.source}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--line)" }}>
            <div className="panel-title" style={{ marginBottom: 8 }}>
              <span>Edge telemetry</span>
              <span className="model-badge">
                <Cpu size={12} /> {health.model}
              </span>
            </div>
            <div className="health-grid">
              <div className="health-stat">
                <Gauge size={14} /> Accuracy <span className="mono">{health.accuracy.toFixed(1)}%</span>
              </div>
              <div className="health-stat">
                <Cpu size={14} /> CPU <span className="mono">{health.cpu}%</span>
              </div>
              <div className="health-stat">
                <MemoryStick size={14} /> Memory <span className="mono">{health.memory}%</span>
              </div>
              <div className="health-stat">
                <Gauge size={14} /> Latency <span className="mono">{health.latency}ms</span>
              </div>
              <div className="health-stat">
                <Flame size={14} /> Device heat <span className="mono">{health.heat}°C</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ForecastCardProps {
  icon: LucideIcon;
  metricKey: MetricKey;
  current: number;
  predicted: number;
}

function ForecastCard({ icon: Icon, metricKey, current, predicted }: ForecastCardProps) {
  const meta = METRICS[metricKey];
  const danger = predicted > meta.threshold;
  return (
    <div className="panel forecast-card">
      <div className="fc-label">
        <Icon size={14} /> {meta.label}
      </div>
      <div className="fc-flow">
        <span className="fc-current mono">
          {current}
          <span className="fc-unit">{meta.unit}</span>
        </span>
        <span className="fc-arrow">→</span>
        <span className={`fc-predicted mono ${danger ? "danger" : ""}`}>
          {predicted.toFixed(1)}
          <span className="fc-unit">{meta.unit}</span>
        </span>
        {danger && <AlertTriangle size={15} className="fc-warn" />}
      </div>
      <div className="fc-sub">
        Current → +15 min AI prediction · threshold {meta.threshold}
        {meta.unit}
      </div>
    </div>
  );
}

interface ActuatorRowProps {
  icon: LucideIcon;
  label: string;
  id: ActuatorId;
  state: ActuatorState;
  onLabel: ActuatorStatus;
  offLabel: ActuatorStatus;
  toggleAuto: (id: ActuatorId) => void;
  manualSet: (id: ActuatorId, status: ActuatorStatus) => void;
}

function ActuatorRow({ icon: Icon, label, id, state, onLabel, offLabel, toggleAuto, manualSet }: ActuatorRowProps) {
  const isOn = state.status === onLabel;
  return (
    <div className="actuator-row">
      <div className="actuator-left">
        <Icon size={16} color="var(--text-muted)" />
        <div>
          <div className="actuator-name">{label}</div>
          <span className={`status-tag ${isOn ? "live" : ""}`}>{state.status}</span>
        </div>
      </div>
      <div className="actuator-controls">
        <div className="mode-toggle">
          <button className={`mode-btn ${state.auto ? "active" : ""}`} onClick={() => !state.auto && toggleAuto(id)}>
            Auto
          </button>
          <button className={`mode-btn ${!state.auto ? "active" : ""}`} onClick={() => state.auto && toggleAuto(id)}>
            Manual
          </button>
        </div>
        <div className="manual-btns">
          <button className={`manual-btn ${isOn ? "on" : ""}`} disabled={state.auto} onClick={() => manualSet(id, onLabel)}>
            {onLabel}
          </button>
          <button className={`manual-btn ${!isOn ? "on" : ""}`} disabled={state.auto} onClick={() => manualSet(id, offLabel)}>
            {offLabel}
          </button>
        </div>
      </div>
    </div>
  );
}