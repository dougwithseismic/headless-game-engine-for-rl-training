import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import type { EffectsState } from '../../types/effects';
import type { EntityState } from '../../types/telemetry';
import { TEAM_COLORS, TEAM_NAMES, weaponFor, shortId } from '../../constants';

export function EntityTooltip({
  effectsRef,
  canvasRef,
}: {
  effectsRef: MutableRefObject<EffectsState>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}) {
  const [entity, setEntity] = useState<EntityState | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const poll = () => {
      const e = effectsRef.current.hoverEntity;
      setEntity(e);
      if (e && canvasRef.current) {
        const rect = canvasRef.current.getBoundingClientRect();
        setPos({
          x: effectsRef.current.mouseCanvasX + rect.left + 14,
          y: effectsRef.current.mouseCanvasY + rect.top - 10,
        });
      }
      rafRef.current = requestAnimationFrame(poll);
    };
    rafRef.current = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(rafRef.current);
  }, [effectsRef, canvasRef]);

  if (!entity) return null;

  const col = TEAM_COLORS[entity.team] || '#fff';
  const hp = Math.max(0, entity.health / entity.max_health);
  const hpC = hp > 0.5 ? 'var(--hp-good)' : hp > 0.25 ? 'var(--hp-mid)' : 'var(--hp-low)';
  const wep = weaponFor(entity.id);
  const spd = entity.velocity
    ? Math.round(Math.sqrt(entity.velocity[0] ** 2 + entity.velocity[1] ** 2))
    : 0;

  return (
    <div className="entity-tooltip" style={{ display: 'block', left: pos.x, top: pos.y }}>
      <div className="tt-name" style={{ color: col }}>
        {shortId(entity.id)} <span className="tt-dim">{TEAM_NAMES[entity.team] || '?'}</span>
      </div>
      <div>
        <span className="tt-dim">hp</span>{' '}
        <span className="tt-hp" style={{ color: hpC }}>
          {Math.round(entity.health)}
        </span>
        <span className="tt-dim">/{Math.round(entity.max_health)}</span>
      </div>
      <div><span className="tt-dim">weapon</span> {wep}</div>
      <div><span className="tt-dim">speed</span> {spd}</div>
      <div>
        <span className="tt-dim">facing</span> {(entity.facing * 180 / Math.PI).toFixed(0)}&deg;
      </div>
    </div>
  );
}
