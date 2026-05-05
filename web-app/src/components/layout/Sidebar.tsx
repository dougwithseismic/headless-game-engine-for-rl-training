import type { ReactNode } from 'react';

export function Sidebar({ children }: { children: ReactNode }) {
  return <div className="sidebar">{children}</div>;
}
