export function apiUrl(host: string, path: string): string {
  if (import.meta.env.DEV) {
    const port = host.split(':')[1] || '3000';
    return `/proxy/${port}${path}`;
  }
  return `http://${host}${path}`;
}

export function wsUrl(host: string, path: string): string {
  if (import.meta.env.DEV) {
    const port = host.split(':')[1] || '3000';
    return `ws://${location.host}/proxy/${port}${path}`;
  }
  return `ws://${host}${path}`;
}
