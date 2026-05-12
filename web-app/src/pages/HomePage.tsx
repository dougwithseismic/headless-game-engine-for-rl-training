import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useDashboardStore } from '../stores/dashboard-store';
import type { ServerEntry } from '../stores/dashboard-store';
import { useDiscoveredSessions } from '../hooks/use-discover';
import { useServerConfig, useServerMatch } from '../hooks/use-server-status';

function ServerCard({ server, onRemove }: { server: ServerEntry; onRemove?: () => void }) {
  const { data: config, isError: configError } = useServerConfig(server.host);
  const { data: match, isError: matchError } = useServerMatch(server.host);

  const online = !configError && !matchError && !!config;

  return (
    <div className={`server-card ${online ? 'online' : 'offline'}`}>
      <div className="server-card-header">
        <div className="server-card-status">
          <span className={`conn-dot${online ? ' live' : ''}`} />
          <span className="server-card-name">{server.name}</span>
          {server.source === 'auto' && (
            <span className="server-card-badge auto">Auto</span>
          )}
        </div>
        {server.source === 'manual' && onRemove && (
          <button className="server-card-remove" onClick={onRemove} title="Remove server">
            &times;
          </button>
        )}
      </div>

      <div className="server-card-host">{server.host}</div>

      {online ? (
        <>
          <div className="server-card-title">{config.title}</div>
          <div className="server-card-stats">
            <div className="server-card-stat">
              <span className="server-card-stat-label">Tick</span>
              <span className="server-card-stat-value">{match?.tick?.toLocaleString() ?? '—'}</span>
            </div>
            <div className="server-card-stat">
              <span className="server-card-stat-label">Rate</span>
              <span className="server-card-stat-value">{config.tick_rate}/s</span>
            </div>
            <div className="server-card-stat">
              <span className="server-card-stat-label">Teams</span>
              <span className="server-card-stat-value">
                {config.teams.count} &times; {config.teams.players_per_team}
              </span>
            </div>
          </div>
          <Link to={`/viewer?host=${encodeURIComponent(server.host)}`} className="server-card-open">
            Open Viewer
          </Link>
        </>
      ) : (
        <div className="server-card-offline">Not responding</div>
      )}
    </div>
  );
}

function AddServerForm({ onAdd }: { onAdd: (name: string, host: string) => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [host, setHost] = useState('localhost:');

  if (!open) {
    return (
      <button className="add-server-btn" onClick={() => setOpen(true)}>
        + Add Server
      </button>
    );
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !host.trim()) return;
    onAdd(name.trim(), host.trim());
    setName('');
    setHost('localhost:');
    setOpen(false);
  };

  return (
    <form className="add-server-form" onSubmit={handleSubmit}>
      <input
        className="add-server-input"
        placeholder="Name"
        value={name}
        onChange={e => setName(e.target.value)}
        autoFocus
      />
      <input
        className="add-server-input host"
        placeholder="localhost:3001"
        value={host}
        onChange={e => setHost(e.target.value)}
      />
      <button type="submit" className="add-server-submit">Add</button>
      <button type="button" className="add-server-cancel" onClick={() => setOpen(false)}>Cancel</button>
    </form>
  );
}

export function HomePage() {
  const manualServers = useDashboardStore(s => s.servers);
  const addServer = useDashboardStore(s => s.addServer);
  const removeServer = useDashboardStore(s => s.removeServer);
  const { data: discovered } = useDiscoveredSessions();

  const merged = useMemo(() => {
    const manualHosts = new Set(manualServers.map(s => s.host));

    const autoEntries: ServerEntry[] = (discovered ?? [])
      .filter(d => !manualHosts.has(`localhost:${d.port}`))
      .map(d => ({
        id: `auto-${d.pid}-${d.port}`,
        name: d.title,
        host: `localhost:${d.port}`,
        source: 'auto' as const,
      }));

    return [...autoEntries, ...manualServers];
  }, [discovered, manualServers]);

  return (
    <>
      <header>
        <div className="header-left">
          <span className="logo">GhostLobby</span>
        </div>
      </header>
      <div className="dashboard">
        <div className="dashboard-header">
          <h1 className="dashboard-title">Sessions</h1>
          <AddServerForm onAdd={addServer} />
        </div>
        <div className="server-grid">
          {merged.map(s => (
            <ServerCard
              key={s.id}
              server={s}
              onRemove={s.source === 'manual' ? () => removeServer(s.id) : undefined}
            />
          ))}
        </div>
        {merged.length === 0 && (
          <div className="dashboard-empty">
            No sessions found. Start a training run or add a server manually.
          </div>
        )}
      </div>
    </>
  );
}
