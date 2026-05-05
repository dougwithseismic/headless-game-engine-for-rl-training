import { useEffect, useRef } from 'react';
import type { MutableRefObject } from 'react';
import type { TelemetryEvent } from '../types/telemetry';
import type { EffectsState } from '../types/effects';
import { useGameStore } from '../stores/game-store';
import { useCameraStore } from '../stores/camera-store';
import { TEAM_COLORS, shortId } from '../constants';
import { spawnParticles, spawnDmgNumber, spawnDecal, spawnRipple } from '../renderer/draw-effects';

export function useWebSocket(effectsRef: MutableRefObject<EffectsState>) {
  const snapCountRef = useRef(0);
  const lastSnapTimeRef = useRef(performance.now());
  const pendingSnapshotRef = useRef<{ tick: number; entities: TelemetryEvent extends { type: 'WorldSnapshot' } ? TelemetryEvent : never } | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let rafId: number;

    const flushToStore = () => {
      const snap = pendingSnapshotRef.current;
      if (snap) {
        useGameStore.getState().processWorldSnapshot(snap.tick, (snap as any).entities);
        pendingSnapshotRef.current = null;
      }
      rafId = requestAnimationFrame(flushToStore);
    };
    rafId = requestAnimationFrame(flushToStore);

    const tpsInterval = setInterval(() => {
      const now = performance.now();
      const elapsed = now - lastSnapTimeRef.current;
      if (elapsed > 0) {
        const tps = Math.round(snapCountRef.current / (elapsed / 1000));
        useGameStore.getState().updateTps(tps);
        snapCountRef.current = 0;
        lastSnapTimeRef.current = now;
      }
    }, 1000);

    function connect() {
      ws = new WebSocket(`ws://${location.host}/ws/observe`);

      ws.onopen = () => {
        useGameStore.getState().setConnected(true);
        useGameStore.getState().addLogEntry(
          '<span class="timestamp">[000000]</span> <span class="event-type">SYS  </span> connected to /ws/observe'
        );
      };

      ws.onclose = () => {
        useGameStore.getState().setConnected(false);
        setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        useGameStore.getState().setConnected(false);
      };

      ws.onmessage = (ev) => {
        try {
          const d: TelemetryEvent = JSON.parse(ev.data);
          const effects = effectsRef.current;

          switch (d.type) {
            case 'WorldSnapshot': {
              snapCountRef.current++;
              for (const e of d.entities) effects.entityIdMap[e.id] = e;
              pendingSnapshotRef.current = d as any;
              break;
            }

            case 'Kill': {
              useGameStore.getState().processKill(d.tick, d.killer, d.victim);
              const victim = effects.entityIdMap[d.victim];
              if (victim) {
                const vCol = TEAM_COLORS[victim.team] || '#fff';
                spawnParticles(effects, victim.position[0], victim.position[1], vCol, 18, 'kill');
                spawnDecal(effects, victim.position[0], victim.position[1], vCol);
                useCameraStore.getState().addShake(1.2);
              }
              break;
            }

            case 'Damage': {
              const target = effects.entityIdMap[d.target];
              if (target) {
                spawnParticles(effects, target.position[0], target.position[1], '#ff6b6b', 5, 'damage');
                spawnDmgNumber(effects, target.position[0], target.position[1], d.amount || 10, '#ff6b6b');
              }
              const ts = String(d.tick).padStart(6, '0');
              useGameStore.getState().addLogEntry(
                `<span class="timestamp">[${ts}]</span> <span class="event-type event-damage">DMG  </span> <span class="hl-id">${shortId(d.target)}</span> <span class="hl-num">-${(d.amount || 0).toFixed(1)}</span>`
              );
              break;
            }

            case 'ShotFired': {
              const range = 500;
              effects.shotTraces.push({
                ox: d.origin[0], oy: d.origin[1],
                ex: d.origin[0] + d.direction[0] * range,
                ey: d.origin[1] + d.direction[1] * range,
                hit: d.hit_target !== null,
                alpha: 1,
              });
              spawnRipple(effects, d.origin[0], d.origin[1]);
              break;
            }

            case 'Spawn': {
              const ts = String(d.tick).padStart(6, '0');
              useGameStore.getState().addLogEntry(
                `<span class="timestamp">[${ts}]</span> <span class="event-type event-spawn">SPAWN</span> <span class="hl-id">${shortId(d.entity || 0)}</span> spawned`
              );
              if (d.position) {
                spawnParticles(effects, d.position[0], d.position[1], '#4ade80', 10, 'spawn');
              }
              break;
            }
          }
        } catch {
          // ignore parse errors
        }
      };
    }

    connect();

    return () => {
      cancelAnimationFrame(rafId);
      clearInterval(tpsInterval);
      ws?.close();
    };
  }, [effectsRef]);
}
