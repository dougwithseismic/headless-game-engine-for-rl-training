import { useQuery } from '@tanstack/react-query';
import type { GameConfig, MatchResponse } from '../types/config';
import { apiUrl } from '../lib/server-url';

export function useServerConfig(host: string) {
  return useQuery<GameConfig>({
    queryKey: ['server-config', host],
    queryFn: () => fetch(apiUrl(host, '/api/config')).then(r => {
      if (!r.ok) throw new Error('unreachable');
      return r.json();
    }),
    refetchInterval: 10_000,
    retry: 0,
  });
}

export function useServerMatch(host: string) {
  return useQuery<MatchResponse>({
    queryKey: ['server-match', host],
    queryFn: () => fetch(apiUrl(host, '/api/match')).then(r => {
      if (!r.ok) throw new Error('unreachable');
      return r.json();
    }),
    refetchInterval: 3_000,
    retry: 0,
  });
}
