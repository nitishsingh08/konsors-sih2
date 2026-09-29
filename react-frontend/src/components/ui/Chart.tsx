/* =====================================================================
   The one place ECharts is registered and mounted. Only the chart types
   and components this dashboard actually draws are pulled in, so the
   bundle stays a fraction of the full library.
   ===================================================================== */
import { useEffect, useRef, type CSSProperties } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, HeatmapChart, LineChart, PieChart, ScatterChart } from 'echarts/charts';
import {
  CalendarComponent,
  DataZoomComponent,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { LabelLayout } from 'echarts/features';
import { SVGRenderer } from 'echarts/renderers';
import { cv, tk } from '../../lib/core';

echarts.use([
  SVGRenderer,
  BarChart, HeatmapChart, LineChart, PieChart, ScatterChart,
  CalendarComponent, DataZoomComponent, GridComponent, MarkAreaComponent, MarkLineComponent,
  TooltipComponent, VisualMapComponent, LabelLayout,
]);

export type ECharts = echarts.ECharts;
/** Option objects are assembled as plain literals; ECharts validates them. */
export type ChartOption = Record<string, unknown>;

export function Chart({
  option,
  height,
  className,
  style,
  onInit,
  ariaLabel,
}: {
  option: ChartOption;
  height?: number | string;
  className?: string;
  style?: CSSProperties;
  onInit?: (c: ECharts) => void;
  ariaLabel?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<ECharts | null>(null);
  const onInitRef = useRef(onInit);
  onInitRef.current = onInit;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chart = echarts.init(el, null, { renderer: 'svg' });
    inst.current = chart;
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(el);
    onInitRef.current?.(chart);
    return () => {
      ro.disconnect();
      inst.current = null;
      chart.dispose();
    };
  }, []);

  useEffect(() => {
    inst.current?.setOption({
      textStyle: { fontFamily: cv('--font') },
      animationDuration: 350,
      ...option,
    } as echarts.EChartsCoreOption);
  }, [option]);

  return <div ref={ref} className={className} role={ariaLabel ? 'img' : undefined} aria-label={ariaLabel} style={{ height, ...style }} />;
}

/** dark tooltip box matching the panel styling */
export const tooltipBox = (): Record<string, unknown> => {
  const T = tk();
  return {
    backgroundColor: T.panel,
    borderColor: T.line2,
    textStyle: { color: T.tx, fontSize: 12 },
    extraCssText: 'box-shadow:0 8px 24px rgba(0,0,0,.3);border-radius:8px',
  };
};

/** the axis chrome every chart reuses */
export const axisBox = (): Record<string, unknown> => {
  const T = tk();
  return {
    axisLine: { lineStyle: { color: T.line2 } },
    axisTick: { show: false },
    axisLabel: { color: T.dim, fontSize: 11 },
    splitLine: { lineStyle: { color: T.line } },
  };
};
