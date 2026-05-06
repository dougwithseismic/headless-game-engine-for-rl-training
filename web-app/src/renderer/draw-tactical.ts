import { toCanvas, scale, type CameraParams } from './math';
import { TEAM_COLORS } from '../constants';
import type { EntityState } from '../types/telemetry';
import type { TacticalState } from '../stores/game-store';

export function drawTacticalOverlay(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  entities: EntityState[],
  tacticalStates: Record<number, TacticalState>,
) {
  const s = scale(cam, canvas, arenaW);

  for (const e of entities) {
    if (e.is_dead) continue;

    const tac = tacticalStates[e.id];
    if (!tac) continue;

    const [ex, ey] = toCanvas(e.position[0], e.position[1], canvas, arenaW, arenaH, cam);
    if (ex < -200 || ex > w + 200 || ey < -200 || ey > h + 200) continue;

    const teamCol = TEAM_COLORS[e.team] || '#fff';

    // --- 0. Vision cone (smooth raycast polygon) ---
    if (tac.rayDistances && tac.rayDistances.length > 0) {
      const numRays = tac.rayDistances.length;
      const sensorRange = 500;

      // Build screen-space points from ray endpoints
      const points: [number, number][] = [];
      for (let i = 0; i < numRays; i++) {
        const angle = (i / numRays) * Math.PI * 2;
        const dist = tac.rayDistances[i] * sensorRange;
        const worldX = e.position[0] + Math.cos(angle) * dist;
        const worldY = e.position[1] + Math.sin(angle) * dist;
        points.push(toCanvas(worldX, worldY, canvas, arenaW, arenaH, cam));
      }

      ctx.save();

      // Filled vision area with radial gradient
      const maxScreenR = sensorRange * s;
      const grad = ctx.createRadialGradient(ex, ey, 0, ex, ey, maxScreenR);
      grad.addColorStop(0, teamCol + '18');
      grad.addColorStop(0.6, teamCol + '0a');
      grad.addColorStop(1, teamCol + '00');

      ctx.fillStyle = grad;
      ctx.beginPath();

      // Catmull-Rom-style smoothing via quadratic curves
      ctx.moveTo(
        (points[points.length - 1][0] + points[0][0]) / 2,
        (points[points.length - 1][1] + points[0][1]) / 2,
      );
      for (let i = 0; i < points.length; i++) {
        const next = points[(i + 1) % points.length];
        const midX = (points[i][0] + next[0]) / 2;
        const midY = (points[i][1] + next[1]) / 2;
        ctx.quadraticCurveTo(points[i][0], points[i][1], midX, midY);
      }
      ctx.closePath();
      ctx.fill();

      // Soft edge outline
      ctx.globalAlpha = 0.12;
      ctx.strokeStyle = teamCol;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      ctx.restore();
    }

    // --- 1. Candidate positions (12 dots) ---
    for (let i = 0; i < tac.candidates.length; i++) {
      const [cx, cy] = tac.candidates[i];
      const [sx, sy] = toCanvas(cx, cy, canvas, arenaW, arenaH, cam);

      // Skip offscreen candidates
      if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) continue;

      const hasLos = tac.candidateLos[i] ?? false;
      const isChosen = i === tac.moveTarget;

      // Distance-based dimming
      const dx = cx - e.position[0], dy = cy - e.position[1];
      const dist = Math.sqrt(dx * dx + dy * dy);
      const distAlpha = Math.max(0.2, 1.0 - dist / 300);

      ctx.globalAlpha = isChosen ? 0.9 : 0.3 * distAlpha;

      // Dot color: green = LOS to enemy, red = in cover
      ctx.fillStyle = hasLos ? '#22c55e' : '#ef4444';
      const dotRadius = isChosen ? Math.max(4, 6 * s / 0.7) : Math.max(2, 4 * s / 0.7);

      ctx.beginPath();
      ctx.arc(sx, sy, dotRadius, 0, Math.PI * 2);
      ctx.fill();

      // Highlight ring on the chosen candidate
      if (isChosen) {
        ctx.strokeStyle = hasLos ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(sx, sy, dotRadius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // --- 2. Dashed line to chosen candidate ---
    if (tac.moveTarget < tac.candidates.length) {
      const chosen = tac.candidates[tac.moveTarget];
      const [sx, sy] = toCanvas(chosen[0], chosen[1], canvas, arenaW, arenaH, cam);

      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = teamCol;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(sx, sy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // --- 3. A* path waypoints ---
    if (tac.path.length > 0) {
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();

      const [firstX, firstY] = toCanvas(tac.path[0][0], tac.path[0][1], canvas, arenaW, arenaH, cam);
      ctx.moveTo(firstX, firstY);

      for (let i = 1; i < tac.path.length; i++) {
        const [px, py] = toCanvas(tac.path[i][0], tac.path[i][1], canvas, arenaW, arenaH, cam);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Small dots at each waypoint
      ctx.fillStyle = '#ffffff';
      ctx.globalAlpha = 0.25;
      const wpRadius = Math.max(1.5, 2 * s / 0.7);
      for (const wp of tac.path) {
        const [px, py] = toCanvas(wp[0], wp[1], canvas, arenaW, arenaH, cam);
        if (px < -10 || px > w + 10 || py < -10 || py > h + 10) continue;
        ctx.beginPath();
        ctx.arc(px, py, wpRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- 4. Aim direction ray ---
    {
      const aimLen = 60;
      const aimEndX = e.position[0] + Math.cos(tac.aimAngle) * aimLen;
      const aimEndY = e.position[1] + Math.sin(tac.aimAngle) * aimLen;
      const [ax, ay] = toCanvas(aimEndX, aimEndY, canvas, arenaW, arenaH, cam);

      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = tac.shooting ? '#ef4444' : '#ffffff';
      ctx.lineWidth = tac.shooting ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ax, ay);
      ctx.stroke();
    }
  }

  ctx.globalAlpha = 1;
}
