import { useState } from 'react';
import { Activity, Users, FileBarChart, Plus } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { RiskGauge } from './components/RiskGauge';
import { MetricCard } from './components/MetricCard';
import { HealthForm } from './components/HealthForm';
import { ResultPanel } from './components/ResultPanel';
import { FeatureBreakdown } from './components/FeatureBreakdown';
import { RecentAssessments } from './components/RecentAssessments';

const initialState = {
  HighBP: 0, HighChol: 0, CholCheck: 1, BMI: 25, Smoker: 0, 
  Stroke: 0, HeartDiseaseorAttack: 0, PhysActivity: 1, Fruits: 1, 
  Veggies: 1, HvyAlcoholConsump: 0, AnyHealthcare: 1, NoDocbcCost: 0, 
  GenHlth: 2, MentHlth: 0, PhysHlth: 0, DiffWalk: 0, Sex: 0, 
  Age: 5, Education: 6, Income: 8
};

function App() {
  const [formData, setFormData] = useState(initialState);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      setResult(data);
      
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      console.error("Failed to fetch prediction:", error);
      alert("Error connecting to DiaSense API. Please check if the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="lg:pl-64 flex flex-col min-h-screen">
        <Header />
        
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full">
          <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Patient Assessment</h1>
              <p className="text-sm text-slate-500 mt-1">Generate real-time diabetes risk predictions using ML.</p>
            </div>
            <button className="hidden sm:flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 shadow-sm transition-colors font-medium text-sm">
              <Plus className="w-4 h-4 mr-2" />
              New Patient
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <MetricCard title="Total Screened" value="2,543" subtitle="+12% from last month" icon={Users} trend="up" />
            <MetricCard title="High Risk Flags" value="482" subtitle="18.9% positivity rate" icon={Activity} trend="neutral" />
            <MetricCard title="Avg Prediction Time" value="124ms" subtitle="Optimized via XGBoost" icon={FileBarChart} trend="down" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            <div className="lg:col-span-2 space-y-8 order-2 lg:order-1">
              <HealthForm 
                formData={formData} 
                setFormData={setFormData} 
                onSubmit={handleSubmit} 
                isLoading={isLoading} 
              />
            </div>

            <div className="space-y-6 order-1 lg:order-2">
              <div className="sticky top-24">
                <RiskGauge 
                  score={result?.risk_score || 0} 
                  level={result?.risk_level || "Unknown"} 
                  percentage={result?.risk_percentage || 0} 
                />
                
                {result && <ResultPanel result={result} />}
                
                <FeatureBreakdown />
                <RecentAssessments />
              </div>
            </div>

          </div>
        </main>
      </div>
      
      <button 
        onClick={() => {
          setFormData(initialState);
          setResult(null);
          window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }}
        className="lg:hidden fixed bottom-6 right-6 p-4 bg-primary-600 text-white rounded-full shadow-lg hover:bg-primary-700 transition-transform active:scale-95 z-50"
      >
        <Plus className="w-6 h-6" />
      </button>
    </div>
  );
}

export default App;
