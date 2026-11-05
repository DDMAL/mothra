export interface CurrentUser {
  id: number;
  username: string;
  email: string;
  firstName: string;
  createdAt: string;
}

// What /api/login and /api/register hand back, camel-cased. refreshToken is
// optional only so a backend predating refresh tokens still logs in -- the
// access token alone then works until it expires, as it always did.
export interface AuthSession {
  user: CurrentUser;
  token: string;
  refreshToken?: string;
}

const TOKEN_KEY = "mothra_token";
const REFRESH_TOKEN_KEY = "mothra_refresh_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(t: string) {
  localStorage.setItem(TOKEN_KEY, t);
}
export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}
export function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}
export function setRefreshToken(t: string) {
  localStorage.setItem(REFRESH_TOKEN_KEY, t);
}
export function clearRefreshToken(): void {
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}
