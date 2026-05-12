import { useQuery } from '@tanstack/react-query';
import { apiUrl } from '../lib/server-url';

export interface AgentInfo {
  type: 'model' | 'scripted';
  label: string;
  detail: string;
}

export interface TrainingInfo {
  model_version: number;
  phase: number | null;
  phase_desc: string;
  last_reload_ago: number;
  steps?: number;
  total_timesteps?: number;
  reward?: number;
  peak_reward?: number;
  n_envs?: number;
  lr?: number;
  reward_history?: { step: number; reward: number }[];
  agents?: Record<string, AgentInfo>;
}

export function useTrainingInfo(host: string) {
  return useQuery<TrainingInfo>({
    queryKey: ['training', host],
    queryFn: () => fetch(apiUrl(host, '/api/training')).then(r => {
      if (!r.ok) throw new Error('No training endpoint');
      return r.json();
    }),
    refetchInterval: 5000,
    retry: 1,
  });
}
