import { create } from 'zustand';

export interface ServerEntry {
  id: string;
  name: string;
  host: string;
  source: 'manual' | 'auto';
}

function load(): ServerEntry[] {
  try {
    const raw = localStorage.getItem('ghostlobby-servers');
    if (raw) {
      const parsed = JSON.parse(raw) as Array<{ id: string; name: string; host: string; source?: string }>;
      return parsed.map(s => ({ ...s, source: (s.source ?? 'manual') as 'manual' | 'auto' }));
    }
  } catch { /* ignore */ }
  return [];
}

function save(servers: ServerEntry[]) {
  localStorage.setItem('ghostlobby-servers', JSON.stringify(servers));
}

interface DashboardState {
  servers: ServerEntry[];
  addServer: (name: string, host: string) => void;
  removeServer: (id: string) => void;
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  servers: load(),
  addServer: (name, host) => {
    const id = crypto.randomUUID();
    const servers = [...get().servers, { id, name, host, source: 'manual' as const }];
    save(servers);
    set({ servers });
  },
  removeServer: (id) => {
    const servers = get().servers.filter(s => s.id !== id);
    save(servers);
    set({ servers });
  },
}));
