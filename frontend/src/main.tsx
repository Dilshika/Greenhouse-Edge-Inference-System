import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import GreenhouseDashboard from "../GreenhouseDashboard";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <GreenhouseDashboard />
  </StrictMode>
);