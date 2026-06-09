import { useState, useEffect, useCallback } from 'react';
import { Upload, Brain, BarChart3, Clock, CheckCircle, XCircle, Loader2, Download } from 'lucide-react';
import { UploadArea } from '../components/UploadArea';
import { TrainingStatus } from '../components/TrainingStatus';
import { TrainingHistory } from '../components/TrainingHistory';
import { ModelInfo } from '../components/ModelInfo';
import { apiFetch } from '../hooks/useApi';

export default function TrainingPage() {
  const [uploadResult, setUploadResult] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState({ status: 'idle', progress: 0, message: '' });
  const [history, setHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const data = await apiFetch('/train/status');
      setTrainingStatus(data);
      setIsTraining(data.status === 'running');
    } catch { }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const data = await apiFetch('/train/history?limit=10');
      setHistory(data);
    } catch { }
  }, []);

  const fetchModelInfo = useCallback(async () => {
    try {
      const data = await apiFetch('/model/info');
      setModelInfo(data);
    } catch { }
  }, []);

  useEffect(() => {
    fetchTrainingStatus();
    fetchHistory();
    fetchModelInfo();
  }, [fetchTrainingStatus, fetchHistory, fetchModelInfo]);

  useEffect(() => {
    if (!isTraining) return;
    const interval = setInterval(() => {
      fetchTrainingStatus();
      fetchHistory();
      if (trainingStatus.status === 'completed' || trainingStatus.status === 'failed') {
        setIsTraining(false);
        fetchModelInfo();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [isTraining, trainingStatus.status, fetchTrainingStatus, fetchHistory, fetchModelInfo]);

  const handleUploadComplete = (result) => {
    setUploadResult(result);
  };

  const handleStartTraining = async () => {
    if (!uploadResult?.filename_key) return;
    try {
      await apiFetch(`/train/start?filename_key=${uploadResult.filename_key}`, { method: 'POST' });
      setIsTraining(true);
      setUploadResult(null);
      fetchTrainingStatus();
    } catch (error) {
      alert(`Failed to start training: ${error.message}`);
    }
  };

  return (
    <>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Model Training</h1>
        <p className="text-sm text-slate-500 mt-1">Upload new patient data to retrain the model. Predictions continue using the current model until the new one is ready.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-1 flex items-center">
            <Upload className="w-5 h-5 mr-2 text-primary-600" />
            Upload Patient Data
          </h2>
          <p className="text-sm text-slate-500 mb-4">Upload a CSV file with the same columns as the original dataset. The file must include a <code className="bg-slate-100 px-1 rounded text-xs">Diabetes_binary</code> column.</p>

          <UploadArea onUploadComplete={handleUploadComplete} disabled={isTraining} />

          {uploadResult && (
            <div className={`mt-4 p-4 rounded-xl border ${uploadResult.valid ? 'bg-success-500/5 border-success-500/30' : 'bg-danger-500/5 border-danger-500/30'}`}>
              <div className="flex items-center mb-2">
                {uploadResult.valid
                  ? <CheckCircle className="w-5 h-5 text-success-600 mr-2" />
                  : <XCircle className="w-5 h-5 text-danger-600 mr-2" />}
                <span className="font-semibold text-sm">{uploadResult.valid ? 'File Valid' : 'Validation Failed'}</span>
              </div>
              <p className="text-sm text-slate-600">{uploadResult.n_rows} rows found ({uploadResult.n_valid_rows} valid)</p>
              {uploadResult.errors && (
                <ul className="text-sm text-danger-600 mt-2 list-disc pl-5">
                  {uploadResult.errors.map((err, i) => <li key={i}>{err}</li>)}
                </ul>
              )}
              {uploadResult.valid && (
                <button
                  onClick={handleStartTraining}
                  disabled={isTraining}
                  className="mt-3 w-full bg-primary-600 hover:bg-primary-700 text-white font-semibold py-2.5 px-4 rounded-xl shadow-sm transition-colors flex items-center justify-center disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <Brain className="w-4 h-4 mr-2" />
                  Start Training
                </button>
              )}
            </div>
          )}
        </div>

        <div className="space-y-6">
          <TrainingStatus status={trainingStatus} />
          <ModelInfo info={modelInfo} />
        </div>
      </div>

      <TrainingHistory history={history} />
    </>
  );
}
