import { describe, it, expect, beforeEach } from 'vitest';
import { useViewerStore } from './viewer-store';

describe('viewer-store', () => {
  beforeEach(() => {
    useViewerStore.setState({
      connected: false,
      tick: 0,
      tps: 0,
      entityCount: 0,
    });
  });

  it('has correct initial state', () => {
    const state = useViewerStore.getState();
    expect(state.connected).toBe(false);
    expect(state.tick).toBe(0);
    expect(state.tps).toBe(0);
    expect(state.entityCount).toBe(0);
  });

  it('updates connection status', () => {
    useViewerStore.getState().setConnected(true);
    expect(useViewerStore.getState().connected).toBe(true);

    useViewerStore.getState().setConnected(false);
    expect(useViewerStore.getState().connected).toBe(false);
  });

  it('updates tick and entity count together', () => {
    useViewerStore.getState().updateTick(42000, 8);
    const state = useViewerStore.getState();
    expect(state.tick).toBe(42000);
    expect(state.entityCount).toBe(8);
  });

  it('updates TPS', () => {
    useViewerStore.getState().updateTps(64);
    expect(useViewerStore.getState().tps).toBe(64);
  });
});
