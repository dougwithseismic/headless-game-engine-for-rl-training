import { useEffect, useRef, useState } from 'react';
import { useTrainingInfo } from '../../hooks/use-training-info';
import { useServerHost } from '../../contexts/server';

function formatAgo(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function formatSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function MiniChart({ history }: { history: { step: number; reward: number }[] }) {
  if (history.length < 2) return null;
  const rewards = history.map(h => h.reward);
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);
  const range = max - min || 1;
  const w = 200;
  const h = 32;
  const points = rewards.map((r, i) => {
    const x = (i / (rewards.length - 1)) * w;
    const y = h - ((r - min) / range) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  const zeroY = max > 0 && min < 0
    ? h - ((0 - min) / range) * (h - 4) - 2
    : null;

  return (
    <svg width={w} height={h} style={{ display: 'block', marginTop: 6 }}>
      {zeroY !== null && (
        <line x1={0} y1={zeroY} x2={w} y2={zeroY}
          stroke="var(--border-subtle)" strokeWidth={1} strokeDasharray="3,3" />
      )}
      <polyline points={points} fill="none"
        stroke="var(--accent-text)" strokeWidth={1.5} strokeLinejoin="round" />
      <circle
        cx={(rewards.length - 1) / (rewards.length - 1) * w}
        cy={h - ((rewards[rewards.length - 1] - min) / range) * (h - 4) - 2}
        r={3} fill="var(--accent-text)" />
    </svg>
  );
}

export function TrainingPanel() {
  const host = useServerHost();
  const { data, isError } = useTrainingInfo(host);
  const [flash, setFlash] = useState(false);
  const prevVersion = useRef(0);

  useEffect(() => {
    if (data && data.model_version > prevVersion.current && prevVersion.current > 0) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 2000);
      return () => clearTimeout(t);
    }
    if (data) prevVersion.current = data.model_version;
  }, [data?.model_version]);

  if (isError || !data) return null;

  const progress = data.steps && data.total_timesteps
    ? Math.min(100, (data.steps / data.total_timesteps) * 100)
    : null;

  return (
    <div className="panel training-panel">
      <div className="panel-header">
        <span className="panel-title">Training</span>
        <span className={`panel-badge ${flash ? 'new-version' : 'live'}`}>
          {flash ? 'NEW MODEL' : `v${data.model_version}`}
        </span>
      </div>

      <div className="training-grid">
        <div className="training-row">
          <span className="training-label">Phase</span>
          <span className="training-value">{data.phase_desc}</span>
        </div>
        <div className="training-row">
          <span className="training-label">Steps</span>
          <span className="training-value mono">
            {data.steps != null ? formatSteps(data.steps) : '—'}
            {data.total_timesteps ? (
              <span className="training-dim"> / {formatSteps(data.total_timesteps)}</span>
            ) : null}
          </span>
        </div>
        {progress !== null && (
          <div className="training-progress-wrap">
            <div className="training-progress-bar">
              <div className="training-progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="training-progress-pct">{progress.toFixed(0)}%</span>
          </div>
        )}
        <div className="training-row">
          <span className="training-label">Reward</span>
          <span className={`training-value mono ${(data.reward ?? 0) >= 0 ? 'positive' : 'negative'}`}>
            {data.reward != null ? (data.reward >= 0 ? '+' : '') + data.reward.toFixed(1) : '—'}
          </span>
        </div>
        <div className="training-row">
          <span className="training-label">Peak</span>
          <span className="training-value mono">
            {data.peak_reward != null ? (data.peak_reward >= 0 ? '+' : '') + data.peak_reward.toFixed(1) : '—'}
          </span>
        </div>
        <div className="training-row">
          <span className="training-label">Last update</span>
          <span className="training-value">{formatAgo(data.last_reload_ago)}</span>
        </div>
        {data.n_envs && (
          <div className="training-row">
            <span className="training-label">Envs</span>
            <span className="training-value mono">{data.n_envs}</span>
          </div>
        )}
      </div>

      {data.reward_history && data.reward_history.length > 2 && (
        <div className="training-chart">
          <MiniChart history={data.reward_history} />
        </div>
      )}
    </div>
  );
}
