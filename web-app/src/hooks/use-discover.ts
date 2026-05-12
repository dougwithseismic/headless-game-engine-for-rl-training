import { useQuery } from '@tanstack/react-query';

export interface DiscoveredSession {
  pid: number;
  port: number;
  title: string;
  config_path: string;
  scenario: string;
  started_at: string;
}

export function useDiscoveredSessions() {
  return useQuery<DiscoveredSession[]>({
    queryKey: ['discover'],
    queryFn: () => fetch('/api/discover').then(r => r.ok ? r.json() : []).catch(() => []),
    refetchInterval: 3_000,
    retry: 0,
  });
}
