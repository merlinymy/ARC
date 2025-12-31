import { useState, useRef, useEffect, type ReactNode } from 'react';
import { HelpCircle } from 'lucide-react';

interface TooltipProps {
  content: string;
  children?: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  showIcon?: boolean;
  iconClassName?: string;
}

export function Tooltip({
  content,
  children,
  position = 'top',
  showIcon = false,
  iconClassName = 'w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [adjustedPosition, setAdjustedPosition] = useState(position);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Adjust position if tooltip would overflow viewport
  useEffect(() => {
    if (isVisible && tooltipRef.current && containerRef.current) {
      const tooltipRect = tooltipRef.current.getBoundingClientRect();

      let newPosition = position;

      // Check if tooltip overflows on top
      if (position === 'top' && tooltipRect.top < 0) {
        newPosition = 'bottom';
      }
      // Check if tooltip overflows on bottom
      else if (position === 'bottom' && tooltipRect.bottom > window.innerHeight) {
        newPosition = 'top';
      }
      // Check if tooltip overflows on left
      else if (position === 'left' && tooltipRect.left < 0) {
        newPosition = 'right';
      }
      // Check if tooltip overflows on right
      else if (position === 'right' && tooltipRect.right > window.innerWidth) {
        newPosition = 'left';
      }

      if (newPosition !== adjustedPosition) {
        setAdjustedPosition(newPosition);
      }
    }
  }, [isVisible, position, adjustedPosition]);

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-gray-800 dark:border-t-gray-700 border-x-transparent border-b-transparent',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-gray-800 dark:border-b-gray-700 border-x-transparent border-t-transparent',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-gray-800 dark:border-l-gray-700 border-y-transparent border-r-transparent',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-gray-800 dark:border-r-gray-700 border-y-transparent border-l-transparent',
  };

  return (
    <div
      ref={containerRef}
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {showIcon ? (
        <button
          type="button"
          className="cursor-help focus:outline-none"
          aria-label="More information"
        >
          <HelpCircle className={iconClassName} />
        </button>
      ) : (
        children
      )}

      {isVisible && (
        <div
          ref={tooltipRef}
          className={`absolute z-50 ${positionClasses[adjustedPosition]} pointer-events-none`}
        >
          <div className="relative">
            <div className="px-3 py-1.5 text-xs leading-relaxed text-white bg-gray-800 dark:bg-gray-700 rounded-lg shadow-lg min-w-48 max-w-sm whitespace-normal">
              {content}
            </div>
            <div
              className={`absolute w-0 h-0 border-4 ${arrowClasses[adjustedPosition]}`}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Inline tooltip that shows next to a label
interface InfoTooltipProps {
  content: string;
  className?: string;
}

export function InfoTooltip({ content, className = '' }: InfoTooltipProps) {
  return (
    <Tooltip content={content} showIcon iconClassName={`w-3.5 h-3.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 cursor-help ${className}`} />
  );
}
