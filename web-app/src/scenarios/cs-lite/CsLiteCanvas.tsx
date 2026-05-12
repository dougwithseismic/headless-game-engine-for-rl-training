import { useEffect } from 'react';
import { CsLite3DCanvas } from './CsLite3DScene';
import { Minimap } from './Minimap';
import { useCsLiteStore, type CsAgent } from './store';
import { useGameConfig } from '../../hooks/use-game-config';
import { useServerHost } from '../../contexts/server';
import { apiUrl } from '../../lib/server-url';

const T_COLOR = '#f06449';
const CT_COLOR = '#4da6ff';
const T_BG = 'rgba(240, 100, 73, 0.12)';
const CT_BG = 'rgba(77, 166, 255, 0.12)';
const T_BORDER = 'rgba(240, 100, 73, 0.3)';
const CT_BORDER = 'rgba(77, 166, 255, 0.3)';

function BottomControls() {
  const cameraMode = useCsLiteStore((s) => s.cameraMode);
  const toggleCamera = useCsLiteStore((s) => s.toggleCameraMode);
  const xray = useCsLiteStore((s) => s.xray);
  const toggleXRay = useCsLiteStore((s) => s.toggleXRay);
  const cameraLabel = cameraMode === 'cinematic' ? 'Cinematic' : 'Free Cam';

  const btnStyle: React.CSSProperties = {
    background: 'rgba(0, 0, 0, 0.75)',
    border: '1px solid #444',
    borderRadius: 4,
    color: '#ccc',
    padding: '4px 14px',
    fontFamily: 'monospace',
    fontSize: 11,
    cursor: 'pointer',
    backdropFilter: 'blur(4px)',
  };

  return (
    <div style={{
      position: 'absolute',
      bottom: 12,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 10,
      display: 'flex',
      gap: 6,
    }}>
      <button onClick={toggleCamera} style={btnStyle}>{cameraLabel}</button>
      <button
        onClick={toggleXRay}
        style={{
          ...btnStyle,
          border: xray ? '1px solid #4da6ff' : '1px solid #444',
          color: xray ? '#4da6ff' : '#666',
        }}
      >
        X-Ray
      </button>
    </div>
  );
}

function PlayerCard({ agent, side }: { agent: CsAgent; side: 'T' | 'CT' }) {
  const color = side === 'T' ? T_COLOR : CT_COLOR;
  const bg = side === 'T' ? T_BG : CT_BG;
  const border = side === 'T' ? T_BORDER : CT_BORDER;
  const hpFrac = agent.health / agent.maxHealth;
  const hpColor = hpFrac > 0.5 ? '#22c55e' : hpFrac > 0.25 ? '#eab308' : '#ef4444';
  const isRight = side === 'CT';

  return (
    <div style={{
      display: 'flex',
      flexDirection: isRight ? 'row-reverse' : 'row',
      alignItems: 'center',
      gap: 6,
      padding: '4px 8px',
      background: agent.isDead ? 'rgba(0, 0, 0, 0.4)' : bg,
      border: `1px solid ${agent.isDead ? 'rgba(255,255,255,0.06)' : border}`,
      borderRadius: 4,
      opacity: agent.isDead ? 0.35 : 1,
      transition: 'opacity 0.6s ease, background 0.4s ease',
      minWidth: 100,
    }}>
      {/* Avatar circle */}
      <div style={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        background: agent.isDead ? '#333' : `${color}22`,
        border: `2px solid ${agent.isDead ? '#444' : color}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        transition: 'border-color 0.4s ease, background 0.4s ease',
      }}>
        <svg width="14" height="14" viewBox="0 0 14 14">
          <circle cx="7" cy="4.5" r="2.5" fill={agent.isDead ? '#555' : color} />
          <path d="M3 12.5 C3 9.5 5 8 7 8 C9 8 11 9.5 11 12.5" fill={agent.isDead ? '#555' : color} />
        </svg>
      </div>

      {/* Info column */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        flex: 1,
        textAlign: isRight ? 'right' : 'left',
      }}>
        <span style={{
          fontFamily: 'monospace',
          fontSize: 10,
          color: agent.isDead ? '#555' : '#ccc',
          letterSpacing: '0.02em',
          transition: 'color 0.4s ease',
        }}>
          {agent.isDead ? 'DEAD' : `${agent.health.toFixed(0)} HP`}
        </span>
        {/* Health bar */}
        <div style={{
          width: '100%',
          height: 3,
          background: 'rgba(255,255,255,0.08)',
          borderRadius: 2,
          overflow: 'hidden',
        }}>
          <div style={{
            width: `${hpFrac * 100}%`,
            height: '100%',
            background: agent.isDead ? '#333' : hpColor,
            borderRadius: 2,
            transition: 'width 0.3s ease, background 0.3s ease',
          }} />
        </div>
      </div>
    </div>
  );
}

function SpectatorBar() {
  const round = useCsLiteStore((s) => s.round);
  const agents = useCsLiteStore((s) => s.agents);

  const tAgents = [...agents.values()].filter((a) => a.team === 0);
  const ctAgents = [...agents.values()].filter((a) => a.team === 1);

  const phaseLabel = round.phase === 'buy_freeze' ? 'BUY' : round.phase === 'active' ? 'LIVE' : 'END';
  const phaseColor = round.phase === 'active' ? '#2ecc71' : round.phase === 'buy_freeze' ? '#f39c12' : '#95a5a6';

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 10,
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'center',
      padding: '6px 12px',
      gap: 8,
      pointerEvents: 'none',
    }}>
      {/* T side cards */}
      <div style={{ display: 'flex', gap: 4, flex: 1, justifyContent: 'flex-end' }}>
        {tAgents.map((agent) => (
          <PlayerCard key={agent.id} agent={agent} side="T" />
        ))}
      </div>

      {/* Center scoreboard */}
      <div style={{
        background: 'rgba(0, 0, 0, 0.8)',
        borderRadius: 6,
        padding: '6px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontFamily: 'monospace',
        backdropFilter: 'blur(6px)',
        border: '1px solid rgba(255,255,255,0.08)',
        flexShrink: 0,
      }}>
        <span style={{ color: T_COLOR, fontWeight: 'bold', fontSize: 22 }}>{round.tScore}</span>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: phaseColor, fontSize: 10, fontWeight: 'bold', letterSpacing: '0.08em' }}>{phaseLabel}</div>
          <div style={{ color: '#666', fontSize: 9 }}>R{round.roundNumber} {round.phaseTimer.toFixed(0)}s</div>
        </div>
        <span style={{ color: CT_COLOR, fontWeight: 'bold', fontSize: 22 }}>{round.ctScore}</span>
      </div>

      {/* CT side cards */}
      <div style={{ display: 'flex', gap: 4, flex: 1, justifyContent: 'flex-start' }}>
        {ctAgents.map((agent) => (
          <PlayerCard key={agent.id} agent={agent} side="CT" />
        ))}
      </div>
    </div>
  );
}

export function CsLiteCanvas() {
  const host = useServerHost();
  const setObstacles = useCsLiteStore((s) => s.setObstacles);

  useEffect(() => {
    fetch(apiUrl(host, '/api/obstacles'))
      .then((r) => r.json())
      .then((data) => {
        if (data.obstacles) {
          setObstacles(data.obstacles, data.spawn_points ?? []);
        }
      })
      .catch(() => {});
  }, [host, setObstacles]);

  return (
    <div className="viewport" style={{ position: 'relative' }}>
      <CsLite3DCanvas />
      <Minimap />
      <SpectatorBar />
      <BottomControls />
    </div>
  );
}
