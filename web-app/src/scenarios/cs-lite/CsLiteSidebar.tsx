import { useCsLiteStore } from './store';

export function CsLiteSidebar() {
  const round = useCsLiteStore((s) => s.round);
  const agents = useCsLiteStore((s) => s.agents);
  const kills = useCsLiteStore((s) => s.kills);

  const tAgents = [...agents.values()].filter((a) => a.team === 0);
  const ctAgents = [...agents.values()].filter((a) => a.team === 1);

  return (
    <div className="sidebar-panel">
      <h3 className="sidebar-title">CS-Lite</h3>

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#f06449', fontWeight: 'bold' }}>T: {round.tScore}</span>
          <span style={{ color: '#aaa' }}>Round {round.roundNumber}</span>
          <span style={{ color: '#4da6ff', fontWeight: 'bold' }}>CT: {round.ctScore}</span>
        </div>
        <div style={{ textAlign: 'center', fontSize: 11, color: '#888' }}>
          {round.phase === 'buy_freeze' ? 'BUY PHASE' : round.phase === 'active' ? 'LIVE' : 'ROUND END'}
          {' — '}{round.phaseTimer.toFixed(0)}s
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <h4 style={{ color: '#f06449', fontSize: 12, marginBottom: 4 }}>Terrorists</h4>
        {tAgents.map((a) => (
          <div key={a.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, opacity: a.isDead ? 0.4 : 1, marginBottom: 2 }}>
            <span>Agent {a.id}</span>
            <span>{a.isDead ? 'DEAD' : `${a.health.toFixed(0)} HP`}</span>
          </div>
        ))}
      </div>

      <div style={{ marginBottom: 12 }}>
        <h4 style={{ color: '#4da6ff', fontSize: 12, marginBottom: 4 }}>Counter-Terrorists</h4>
        {ctAgents.map((a) => (
          <div key={a.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, opacity: a.isDead ? 0.4 : 1, marginBottom: 2 }}>
            <span>Agent {a.id}</span>
            <span>{a.isDead ? 'DEAD' : `${a.health.toFixed(0)} HP`}</span>
          </div>
        ))}
      </div>

      <div>
        <h4 style={{ color: '#ccc', fontSize: 12, marginBottom: 4 }}>Kill Feed</h4>
        {kills.slice(-8).reverse().map((k, i) => (
          <div key={i} style={{ fontSize: 10, color: '#888', marginBottom: 1 }}>
            [{k.tick}] {k.killer} → {k.victim}
          </div>
        ))}
        {kills.length === 0 && <div style={{ fontSize: 10, color: '#555' }}>No kills yet</div>}
      </div>
    </div>
  );
}
