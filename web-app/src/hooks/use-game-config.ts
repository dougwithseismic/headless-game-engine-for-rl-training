import { useQuery } from '@tanstack/react-query';
import type { GameConfig } from '../types/config';
import { apiUrl } from '../lib/server-url';

export function useGameConfig(host: string) {
  return useQuery<GameConfig>({
    queryKey: ['config', host],
    queryFn: () => fetch(apiUrl(host, '/api/config')).then(r => r.json()),
    staleTime: Infinity,
    retry: 3,
  });
}
