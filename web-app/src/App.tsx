import { Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { ViewerPage } from './pages/ViewerPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/viewer" element={<ViewerPage />} />
    </Routes>
  );
}
