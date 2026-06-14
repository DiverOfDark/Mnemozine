/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional static bearer token (API_CONTRACT.md §Auth). Usually unset (localhost). */
  readonly VITE_MNEMOZINE_TOKEN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
