import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'leaflet/dist/leaflet.css';
import './styles/base.css';
import './styles/analysis.css';
import { App } from './App';

const host = document.getElementById('app');
if (!host) throw new Error('#app is missing from index.html');

createRoot(host).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
