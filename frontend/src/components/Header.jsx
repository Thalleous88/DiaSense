import { Bell, Search, Menu } from "lucide-react";

export function Header() {
  return (
    <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-10 lg:pl-64 flex items-center justify-between px-4 sm:px-6">
      <div className="flex items-center flex-1">
        <button className="lg:hidden p-2 -ml-2 mr-2 text-slate-600 hover:bg-slate-100 rounded-lg">
          <Menu className="w-5 h-5" />
        </button>
        <div className="max-w-md w-full relative hidden sm:block">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-slate-400" />
          </div>
          <input 
            type="text" 
            className="block w-full pl-10 pr-3 py-2 border border-slate-200 rounded-lg leading-5 bg-slate-50 placeholder-slate-400 focus:outline-none focus:bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm transition-colors" 
            placeholder="Search patients or ID..." 
          />
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <button className="relative p-2 text-slate-400 hover:text-slate-500 hover:bg-slate-100 rounded-full transition-colors">
          <span className="absolute top-1.5 right-1.5 block h-2 w-2 rounded-full bg-danger-500 ring-2 ring-white"></span>
          <Bell className="h-5 w-5" />
        </button>
        
        <div className="flex items-center gap-3 border-l border-slate-200 pl-4">
          <img 
            className="h-8 w-8 rounded-full bg-slate-200 object-cover" 
            src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix&backgroundColor=e0f2fe" 
            alt="Dr. Felix" 
          />
          <div className="hidden md:block text-sm">
            <p className="font-medium text-slate-900 leading-none">Dr. Felix</p>
            <p className="text-slate-500 mt-1 text-xs">Endocrinologist</p>
          </div>
        </div>
      </div>
    </header>
  );
}
