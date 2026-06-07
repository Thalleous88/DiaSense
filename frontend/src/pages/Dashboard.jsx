import { useState, useEffect, useCallback } from 'react';
import { Activity, Users, FileBarChart, Plus } from 'lucide-react';
import { RiskGauge } from '../components/RiskGauge';
import { MetricCard } from '../components/MetricCard';
import { HealthForm } from '../components/HealthForm';
import { ResultPanel } from '../components/ResultPanel';
import { FeatureBreakdown } from '../components/FeatureBreakdown';
import { RecentAssessments } from '../components/RecentAssessments';
import { apiFetch } from '../hooks/useApi';

const initialState = {
  HighBP: 0, HighChol: 0, CholCheck: 1, BMI: 25, Smoker: 0,
  Stroke: 0, HeartDiseaseorAttack: 0, PhysActivity: 1, Fruits: 1,
  Veggies: 1, HvyAlcoholConsump: 0, AnyHealthcare: 1, NoDocbcCost: 0,
  GenHlth: 2, MentHlth: 0, PhysHlth: 0, DiffWalk: 0, Sex: 0,
  Age: 5, Education: 6, Income: 8
};

export default function Dashboard() {
  const [formData, setFormData] = useState(initialState);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const data = await apiFetch('/stats');
      setStats(data);
    } catch { }
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const data = await apiFetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      setResult(data);
      fetchStats();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      alert("Error connecting to DiaSense API. Please check if the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewPatient = () => {
    setFormData(initialState);
    setResult(null);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  };

  return (
    <>
      <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Patient Assessment</h1>
          <p className="text-sm text-slate-500 mt-1">Generate real-time diabetes risk predictions using an ML ensemble.</p>
        </div>
        <button onClick={handleNewPatient} className="hidden sm:flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 shadow-sm transition-colors font-medium text-sm">
          <Plus className="w-4 h-4 mr-2" />
          New Patient
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <MetricCard title="Total Screened" value={stats?.total_screened ?? "—"} subtitle={stats?.screened_subtitle ?? "Loading..."} icon={Users} trend="up" />
        <MetricCard title="High Risk Flags" value={stats?.high_risk_count ?? "—"} subtitle={stats?.high_risk_subtitle ?? "Loading..."} icon={Activity} trend="neutral" />
        <MetricCard title="Avg Prediction Time" value={stats?.avg_time ?? "—"} subtitle="Real-time inference" icon={FileBarChart} trend="down" />
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
          <div className="sticky top-20">
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

      <button
        onClick={handleNewPatient}
        className="lg:hidden fixed bottom-6 right-6 p-4 bg-primary-600 text-white rounded-full shadow-lg hover:bg-primary-700 transition-transform active:scale-95 z-50"
      >
        <Plus className="w-6 h-6" />
      </button>
    </>
  );
}
