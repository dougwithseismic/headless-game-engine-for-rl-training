import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useDashboardStore } from './dashboard-store';

describe('dashboard-store', () => {
  beforeEach(() => {
    localStorage.clear();
    useDashboardStore.setState({ servers: [] });
  });

  it('starts with empty servers when localStorage is empty', () => {
    const state = useDashboardStore.getState();
    expect(state.servers).toEqual([]);
  });

  it('adds a manual server', () => {
    const { addServer } = useDashboardStore.getState();
    addServer('Test Server', 'localhost:4000');

    const { servers } = useDashboardStore.getState();
    expect(servers).toHaveLength(1);
    expect(servers[0].name).toBe('Test Server');
    expect(servers[0].host).toBe('localhost:4000');
    expect(servers[0].source).toBe('manual');
    expect(servers[0].id).toBeTruthy();
  });

  it('removes a server by id', () => {
    const { addServer } = useDashboardStore.getState();
    addServer('A', 'localhost:3000');
    addServer('B', 'localhost:3001');

    const { servers } = useDashboardStore.getState();
    expect(servers).toHaveLength(2);

    useDashboardStore.getState().removeServer(servers[0].id);
    expect(useDashboardStore.getState().servers).toHaveLength(1);
    expect(useDashboardStore.getState().servers[0].name).toBe('B');
  });

  it('persists to localStorage', () => {
    useDashboardStore.getState().addServer('Persisted', 'localhost:5000');

    const raw = localStorage.getItem('ghostlobby-servers');
    expect(raw).toBeTruthy();
    const parsed = JSON.parse(raw!);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].name).toBe('Persisted');
  });
});
