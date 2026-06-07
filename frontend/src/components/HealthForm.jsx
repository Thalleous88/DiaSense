export function HealthForm({ formData, setFormData, onSubmit, isLoading }) {
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let parsedValue = value;
    if (type === "checkbox") {
      parsedValue = checked ? 1 : 0;
    } else if (type === "number" || type === "range") {
      parsedValue = parseInt(value, 10);
    } else if (type === "radio") {
      parsedValue = parseInt(value, 10);
    }
    
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };

  const binaryFields = [
    { name: "HighBP", label: "Diagnosed with High Blood Pressure" },
    { name: "HighChol", label: "Diagnosed with High Cholesterol" },
    { name: "CholCheck", label: "Cholesterol Checked in Past 5 Years" },
    { name: "Smoker", label: "Smoked at Least 100 Cigarettes in Lifetime" },
    { name: "Stroke", label: "Ever Diagnosed with Stroke" },
    { name: "HeartDiseaseorAttack", label: "Ever Diagnosed with Coronary Heart Disease or Myocardial Infarction" },
    { name: "PhysActivity", label: "Physical Activity or Exercise in Past 30 Days (Outside Regular Job)" },
    { name: "Fruits", label: "Consume Fruit at Least Once Per Day" },
    { name: "Veggies", label: "Consume Vegetables at Least Once Per Day" },
    { name: "HvyAlcoholConsump", label: "Heavy Alcohol Consumption (Men: >14 drinks/week; Women: >7 drinks/week)" },
    { name: "AnyHealthcare", label: "Has Health Insurance or Healthcare Coverage" },
    { name: "NoDocbcCost", label: "Could Not See Doctor in Past 12 Months Due to Cost" },
    { name: "DiffWalk", label: "Serious Difficulty Walking or Climbing Stairs" },
  ];

  return (
    <div className="glass-card p-6">
      <div className="mb-6 pb-4 border-b border-slate-100">
        <h2 className="text-xl font-semibold text-slate-900">Patient Health Survey</h2>
        <p className="text-sm text-slate-500 mt-1">Complete all fields to generate a diabetes risk prediction score.</p>
      </div>

      <form onSubmit={onSubmit} className="space-y-8">
        
        <div>
          <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wider mb-4">Core Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Body Mass Index (BMI)</label>
              <input 
                type="number" name="BMI" value={formData.BMI} onChange={handleChange}
                min="10" max="99" required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              />
            </div>
            
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Age Category</label>
              <select 
                name="Age" value={formData.Age} onChange={handleChange} required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none bg-white"
              >
                <option value="1">18-24</option>
                <option value="2">25-29</option>
                <option value="3">30-34</option>
                <option value="4">35-39</option>
                <option value="5">40-44</option>
                <option value="6">45-49</option>
                <option value="7">50-54</option>
                <option value="8">55-59</option>
                <option value="9">60-64</option>
                <option value="10">65-69</option>
                <option value="11">70-74</option>
                <option value="12">75-79</option>
                <option value="13">80 or older</option>
              </select>
            </div>

            <div className="space-y-2 col-span-1 md:col-span-2 mt-2">
              <label className="text-sm font-medium text-slate-700 block">Biological Sex</label>
              <div className="flex gap-4">
                <label className="flex items-center p-3 border border-slate-200 rounded-lg flex-1 cursor-pointer hover:bg-slate-50">
                  <input type="radio" name="Sex" value="0" checked={formData.Sex === 0} onChange={handleChange} className="text-primary-600" />
                  <span className="ml-2 text-sm text-slate-700">Female</span>
                </label>
                <label className="flex items-center p-3 border border-slate-200 rounded-lg flex-1 cursor-pointer hover:bg-slate-50">
                  <input type="radio" name="Sex" value="1" checked={formData.Sex === 1} onChange={handleChange} className="text-primary-600" />
                  <span className="ml-2 text-sm text-slate-700">Male</span>
                </label>
              </div>
            </div>

          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wider mb-4">Health Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">General Health (1=Excellent, 5=Poor)</label>
              <input 
                type="range" name="GenHlth" min="1" max="5" value={formData.GenHlth} onChange={handleChange}
                className="w-full accent-primary-600"
              />
              <div className="flex justify-between text-xs text-slate-400">
                <span>Excellent</span>
                <span>Poor</span>
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Days of Poor Physical Health (past 30 days)</label>
              <input 
                type="number" name="PhysHlth" value={formData.PhysHlth} onChange={handleChange}
                min="0" max="30" required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              />
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Days of Poor Mental Health (past 30 days)</label>
              <input 
                type="number" name="MentHlth" value={formData.MentHlth} onChange={handleChange}
                min="0" max="30" required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none"
              />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wider mb-4">Risk Factors & Lifestyle</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-y-3 gap-x-6">
            {binaryFields.map(field => (
              <label key={field.name} className="flex items-start space-x-3 p-2 hover:bg-slate-50 rounded-lg cursor-pointer">
                <input 
                  type="checkbox" 
                  name={field.name}
                  checked={formData[field.name] === 1}
                  onChange={handleChange}
                  className="w-4 h-4 text-primary-600 border-slate-300 rounded focus:ring-primary-500 mt-0.5"
                />
                <span className="text-sm text-slate-700 leading-snug">{field.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-primary-600 uppercase tracking-wider mb-4">Socioeconomic</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Education Level</label>
              <select 
                name="Education" value={formData.Education} onChange={handleChange} required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none bg-white"
              >
                <option value="1">Never attended / Kindergarten</option>
                <option value="2">Grades 1-8 (Elementary)</option>
                <option value="3">Grades 9-11 (Some High School)</option>
                <option value="4">Grade 12 or GED (High School Grad)</option>
                <option value="5">College 1-3 years (Some College)</option>
                <option value="6">College 4+ years (College Grad)</option>
              </select>
            </div>
            
            <div className="space-y-1">
              <label className="text-sm font-medium text-slate-700">Income Level</label>
              <select 
                name="Income" value={formData.Income} onChange={handleChange} required
                className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 outline-none bg-white"
              >
                <option value="1">Less than $10,000</option>
                <option value="2">$10,000 to less than $15,000</option>
                <option value="3">$15,000 to less than $20,000</option>
                <option value="4">$20,000 to less than $25,000</option>
                <option value="5">$25,000 to less than $35,000</option>
                <option value="6">$35,000 to less than $50,000</option>
                <option value="7">$50,000 to less than $75,000</option>
                <option value="8">$75,000 or more</option>
              </select>
            </div>
          </div>
        </div>

        <div className="pt-4">
          <button 
            type="submit" 
            disabled={isLoading}
            className="w-full bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 px-6 rounded-xl shadow-sm hover:shadow-md transition-all flex items-center justify-center disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </span>
            ) : (
              "Generate Risk Assessment"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
