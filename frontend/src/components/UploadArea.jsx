import { useState, useCallback } from 'react';
import { Upload, FileText } from 'lucide-react';
import { apiFetch } from '../hooks/useApi';

export function UploadArea({ onUploadComplete, disabled = false }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [fileName, setFileName] = useState(null);

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      alert('Please upload a CSV file');
      return;
    }
    setFileName(file.name);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const result = await apiFetch('/train/upload', {
        method: 'POST',
        body: formData,
      });
      onUploadComplete(result);
    } catch (error) {
      alert(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [onUploadComplete]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
        isDragging ? 'border-primary-500 bg-primary-50' :
        disabled ? 'border-slate-200 bg-slate-50 cursor-not-allowed' :
        'border-slate-300 hover:border-primary-400 hover:bg-slate-50'
      }`}
      onClick={() => { if (!disabled) document.getElementById('csv-upload-input').click(); }}
    >
      <input
        id="csv-upload-input"
        type="file"
        accept=".csv"
        onChange={handleChange}
        className="hidden"
        disabled={disabled}
      />
      {isUploading ? (
        <div className="flex flex-col items-center">
          <Loader2 className="w-10 h-10 text-primary-500 animate-spin mb-3" />
          <p className="text-sm text-slate-600">Uploading {fileName}...</p>
        </div>
      ) : fileName ? (
        <div className="flex flex-col items-center">
          <FileText className="w-10 h-10 text-primary-500 mb-3" />
          <p className="text-sm font-medium text-slate-700">{fileName}</p>
          <p className="text-xs text-slate-400 mt-1">Drop another file to replace</p>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <Upload className="w-10 h-10 text-slate-400 mb-3" />
          <p className="text-sm font-medium text-slate-700">Drag & drop a CSV file here</p>
          <p className="text-xs text-slate-400 mt-1">or click to browse</p>
        </div>
      )}
      <div className="mt-4 text-left">
        <p className="text-xs font-semibold text-slate-500 mb-1">Required columns:</p>
        <p className="text-xs text-slate-400 leading-relaxed">
          Diabetes_binary, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
          HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
          AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex,
          Age, Education, Income
        </p>
      </div>
    </div>
  );
}

function Loader2(props) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
  );
}
