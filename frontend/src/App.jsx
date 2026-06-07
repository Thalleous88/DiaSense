import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import Dashboard from './pages/Dashboard';
import TrainingPage from './pages/TrainingPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background">
        <Header />
        <main className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<TrainingPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
