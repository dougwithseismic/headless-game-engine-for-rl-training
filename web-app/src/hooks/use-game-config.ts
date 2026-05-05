import { useQuery } from '@tanstack/react-query';
import type { GameConfig } from '../types/config';

export function useGameConfig() {
  return useQuery<GameConfig>({
    queryKey: ['config'],
    queryFn: () => fetch('/api/config').then(r => r.json()),
    staleTime: Infinity,
    retry: 3,
  });
}
