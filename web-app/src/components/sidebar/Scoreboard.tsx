import { useGameStore } from '../../stores/game-store';

export function Scoreboard() {
  const score = useGameStore(s => s.score);

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Scoreboard</span>
        <span className="panel-badge live">LIVE</span>
      </div>
      <div className="score-row">
        <div className="score-team">
          <div className="team-dot" style={{ background: 'var(--team-blue)' }} />
          <span style={{ color: 'var(--team-blue)' }}>Blue</span>
        </div>
        <span className="score-value" style={{ color: 'var(--team-blue)' }}>{score[0] || 0}</span>
      </div>
      <div className="score-row">
        <div className="score-team">
          <div className="team-dot" style={{ background: 'var(--team-red)' }} />
          <span style={{ color: 'var(--team-red)' }}>Red</span>
        </div>
        <span className="score-value" style={{ color: 'var(--team-red)' }}>{score[1] || 0}</span>
      </div>
    </div>
  );
}
