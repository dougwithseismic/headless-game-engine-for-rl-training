import { describe, it, expect, vi } from 'vitest';
import { apiUrl, wsUrl } from './server-url';

describe('apiUrl', () => {
  it('extracts port and builds proxy path in dev', () => {
    vi.stubEnv('DEV', 'true');
    expect(apiUrl('localhost:3000', '/api/config')).toBe('/proxy/3000/api/config');
    expect(apiUrl('localhost:3001', '/api/match')).toBe('/proxy/3001/api/match');
    vi.unstubAllEnvs();
  });

  it('defaults to port 3000 when no port in host', () => {
    vi.stubEnv('DEV', 'true');
    expect(apiUrl('localhost', '/api/config')).toBe('/proxy/3000/api/config');
    vi.unstubAllEnvs();
  });

  it('builds absolute URL in production', () => {
    vi.stubEnv('DEV', '');
    expect(apiUrl('192.168.1.5:4000', '/api/config')).toBe('http://192.168.1.5:4000/api/config');
    vi.unstubAllEnvs();
  });
});

describe('wsUrl', () => {
  it('builds proxy WebSocket URL in dev', () => {
    vi.stubEnv('DEV', 'true');
    const url = wsUrl('localhost:3000', '/ws/observe');
    expect(url).toContain('/proxy/3000/ws/observe');
  });

  it('builds direct WebSocket URL in production', () => {
    vi.stubEnv('DEV', '');
    expect(wsUrl('localhost:3000', '/ws/observe')).toBe('ws://localhost:3000/ws/observe');
    vi.unstubAllEnvs();
  });
});
