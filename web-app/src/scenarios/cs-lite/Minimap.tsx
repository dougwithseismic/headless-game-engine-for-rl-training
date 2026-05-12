import { useRef, useEffect, useCallback } from 'react';
import { useCsLiteStore } from './store';
import { useGameConfig } from '../../hooks/use-game-config';
import { useServerHost } from '../../contexts/server';

const T_COLOR = '#f06449';
const CT_COLOR = '#4da6ff';
const OBSTACLE_COLOR = '#5a5a7a';
const BG_COLOR = 'rgba(12, 12, 26, 0.88)';
const BORDER_COLOR = '#3a3a55';

export function Minimap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const host = useServerHost();
  const { data: config } = useGameConfig(host);
  const agents = useCsLiteStore((s) => s.agents);
  const round = useCsLiteStore((s) => s.round);
  const obstacles = useCsLiteStore((s) => s.obstacles);

  const arenaW = (config?.extra?.arena_width as number) ?? 80;
  const arenaD = (config?.extra?.arena_depth as number) ?? 60;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 200;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = (size * arenaD / arenaW) * dpr;
    ctx.scale(dpr, dpr);

    const w = size;
    const h = size * arenaD / arenaW;
    const pad = 4;
    const scale = (w - pad * 2) / arenaW;

    const toScreen = (x: number, z: number): [number, number] => [
      pad + x * scale,
      pad + z * scale,
    ];

    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = BORDER_COLOR;
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);

    // Bomb sites
    for (const [cx, cz, label] of [
      [arenaW * 0.25, arenaD * 0.75, 'A'],
      [arenaW * 0.75, arenaD * 0.75, 'B'],
    ] as [number, number, string][]) {
      const [sx, sz] = toScreen(cx, cz);
      ctx.fillStyle = 'rgba(231, 76, 60, 0.12)';
      ctx.beginPath();
      ctx.arc(sx, sz, 6 * scale, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(231, 76, 60, 0.5)';
      ctx.font = 'bold 8px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, sx, sz);
    }

    // Obstacles
    ctx.fillStyle = OBSTACLE_COLOR;
    for (const obs of obstacles) {
      const [ox, oz] = toScreen(obs.x, obs.y);
      ctx.fillRect(ox, oz, obs.width * scale, obs.height * scale);
    }

    // Agents
    for (const agent of agents.values()) {
      const [ax, az] = toScreen(agent.x, agent.z);
      const isT = agent.team === 0;

      if (agent.isDead) {
        ctx.fillStyle = isT ? '#5a2020' : '#1a3050';
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.arc(ax, az, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        continue;
      }

      ctx.fillStyle = isT ? T_COLOR : CT_COLOR;
      ctx.beginPath();
      ctx.arc(ax, az, 3, 0, Math.PI * 2);
      ctx.fill();

      // Facing
      const fLen = 6;
      ctx.strokeStyle = isT ? T_COLOR : CT_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ax, az);
      ctx.lineTo(ax + Math.cos(agent.yaw) * fLen, az + Math.sin(agent.yaw) * fLen);
      ctx.stroke();
    }
  }, [agents, obstacles, arenaW, arenaD]);

  useEffect(() => {
    const raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        bottom: 12,
        left: 12,
        width: 200,
        height: 200 * arenaD / arenaW,
        borderRadius: 4,
        border: '1px solid #333',
        zIndex: 10,
        pointerEvents: 'none',
      }}
    />
  );
}
