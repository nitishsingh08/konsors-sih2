import { HashRouter } from 'react-router-dom';
import { ThemeProvider } from './state/ThemeProvider';
import { ReviewsProvider } from './state/ReviewsProvider';
import { ToastProvider } from './state/ToastProvider';
import { TooltipProvider } from './components/ui/Tooltip';
import { Shell } from './components/Shell';

export function App() {
  return (
    <ThemeProvider>
      <ReviewsProvider>
        <ToastProvider>
          <TooltipProvider>
            <HashRouter>
              <Shell />
            </HashRouter>
          </TooltipProvider>
        </ToastProvider>
      </ReviewsProvider>
    </ThemeProvider>
  );
}
