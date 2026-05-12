import { useEffect, useRef } from 'react';
import type { TelemetryEvent } from '../types/telemetry';
import type { ScenarioDefinition } from '../scenarios/types';
import { useViewerStore } from '../stores/viewer-store';
import { wsUrl } from '../lib/server-url';

export function useViewerWebSocket(host: string, scenario: ScenarioDefinition, enabled: boolean) {
  const scenarioRef = useRef(scenario);
  scenarioRef.current = scenario;

  useEffect(() => {
    if (!enabled) return;

    let ws: WebSocket;
    let disposed = false;
    let snapCount = 0;
    let lastSnapTime = performance.now();

    const tpsInterval = setInterval(() => {
      const now = performance.now();
      const elapsed = now - lastSnapTime;
      if (elapsed > 0) {
        useViewerStore.getState().updateTps(Math.round(snapCount / (elapsed / 1000)));
        snapCount = 0;
        lastSnapTime = now;
      }
    }, 1000);

    function connect() {
      ws = new WebSocket(wsUrl(host, '/ws/observe'));

      ws.onopen = () => {
        useViewerStore.getState().setConnected(true);
        scenarioRef.current.onConnect?.();
      };

      ws.onclose = () => {
        useViewerStore.getState().setConnected(false);
        scenarioRef.current.onDisconnect?.();
        if (!disposed) setTimeout(connect, 1000);
      };

      ws.onerror = () => {
        useViewerStore.getState().setConnected(false);
      };

      ws.onmessage = (ev) => {
        try {
          const event: TelemetryEvent = JSON.parse(ev.data);

          if (event.type === 'WorldSnapshot') {
            snapCount++;
            useViewerStore.getState().updateTick(event.tick, event.entities.length);
          }

          scenarioRef.current.onTelemetryEvent(event);
        } catch {
          // ignore parse errors
        }
      };
    }

    connect();

    return () => {
      disposed = true;
      clearInterval(tpsInterval);
      ws?.close();
    };
  }, [host, enabled]);
}
