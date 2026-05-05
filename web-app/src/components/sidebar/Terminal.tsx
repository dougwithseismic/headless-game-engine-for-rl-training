import { useState, useRef, useEffect } from 'react';
import { useGameStore } from '../../stores/game-store';
import { TEAM_NAMES } from '../../constants';

type Tab = 'events' | 'state' | 'perf';

export function Terminal() {
  const [activeTab, setActiveTab] = useState<Tab>('events');
  const eventLog = useGameStore(s => s.eventLog);
  const entities = useGameStore(s => s.entities);
  const tick = useGameStore(s => s.tick);
  const score = useGameStore(s => s.score);
  const tps = useGameStore(s => s.tps);

  return (
    <div className="terminal">
      <div className="terminal-header">
        <div className="terminal-tabs">
          {(['events', 'state', 'perf'] as Tab[]).map(tab => (
            <button
              key={tab}
              className={`terminal-tab${activeTab === tab ? ' active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
        <span style={{ fontSize: '8px', color: 'var(--text-muted)' }}>{eventLog.length}</span>
      </div>

      {activeTab === 'events' && <EventsView eventLog={eventLog} />}
      {activeTab === 'state' && <StateView entities={entities} tick={tick} score={score} />}
      {activeTab === 'perf' && <PerfView tps={tps} entities={entities} />}
    </div>
  );
}

function EventsView({ eventLog }: { eventLog: string[] }) {
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
  }, [eventLog.length]);

  return (
    <div className="terminal-body" ref={bodyRef}>
      {eventLog.slice(0, 60).map((line, i) => (
        <div key={i} className="log-line" dangerouslySetInnerHTML={{ __html: line }} />
      ))}
    </div>
  );
}

function StateView({ entities, tick, score }: { entities: any[]; tick: number; score: number[] }) {
  const alive = entities.filter(e => !e.is_dead);
  const teams: Record<string, { alive: number; dead: number; hp: number }> = {};
  for (const e of entities) {
    const t = TEAM_NAMES[e.team] || 'T' + e.team;
    if (!teams[t]) teams[t] = { alive: 0, dead: 0, hp: 0 };
    if (e.is_dead) teams[t].dead++;
    else { teams[t].alive++; teams[t].hp += Math.round(e.health); }
  }

  return (
    <div className="terminal-body">
      <div className="log-line"><span className="hl-bracket">{'{'}</span></div>
      <div className="log-line">  <span className="hl-key">"tick"</span>: <span className="hl-num">{tick}</span>,</div>
      <div className="log-line">  <span className="hl-key">"alive"</span>: <span className="hl-num">{alive.length}</span> / <span className="hl-num">{entities.length}</span>,</div>
      <div className="log-line">  <span className="hl-key">"score"</span>: [<span className="hl-num">{score[0]}</span>, <span className="hl-num">{score[1]}</span>],</div>
      <div className="log-line">  <span className="hl-key">"teams"</span>: <span className="hl-bracket">{'{'}</span></div>
      {Object.entries(teams).map(([name, data]) => (
        <div key={name} className="log-line">
          {'    '}<span className="hl-str">"{name}"</span>: {'{ '}
          <span className="hl-key">alive</span>: <span className="hl-num">{data.alive}</span>,{' '}
          <span className="hl-key">dead</span>: <span className="hl-num">{data.dead}</span>,{' '}
          <span className="hl-key">hp</span>: <span className="hl-num">{data.hp}</span>
          {' }'}
        </div>
      ))}
      <div className="log-line">  <span className="hl-bracket">{'}'}</span></div>
      <div className="log-line"><span className="hl-bracket">{'}'}</span></div>
    </div>
  );
}

function PerfView({ tps, entities }: { tps: number; entities: any[] }) {
  const fpsEl = document.getElementById('fps-value');
  const fps = fpsEl?.textContent || '—';

  return (
    <div className="terminal-body">
      <div className="log-line"><span className="hl-key">{'server_tps  '}</span> <span className="hl-num">{tps}</span></div>
      <div className="log-line"><span className="hl-key">{'client_fps  '}</span> <span className="hl-num">{fps}</span></div>
      <div className="log-line"><span className="hl-key">{'entities    '}</span> <span className="hl-num">{entities.length}</span></div>
    </div>
  );
}
