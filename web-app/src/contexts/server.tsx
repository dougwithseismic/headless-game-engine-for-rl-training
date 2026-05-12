import { createContext, useContext } from 'react';

const ServerHostContext = createContext('localhost:3000');

export const ServerHostProvider = ServerHostContext.Provider;
export const useServerHost = () => useContext(ServerHostContext);
