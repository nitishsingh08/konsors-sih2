/* =====================================================================
   Methods and limits: what each label means and what this build cannot
   do. Port of pageMethods() in frontend/js/pages.js.
   ===================================================================== */
import { CLS } from '../data/classes';
import { Card } from '../components/ui/Primitives';
import { Glyph } from '../components/ui/Glyph';
import { PageHead } from './PageHead';

const IMPLEMENTED = [
  'Command map with filters, layers, hex density, time scrubber and area drawing',
  'Priority queue, KPI strip and place reports with explanations, timelines and baselines',
  'Regional analytics, watchlist, and model health pages',
  'Exports of events as CSV and GeoJSON',
];

const PROPOSED = [
  'Live connection to the FIRMS pipeline, PostGIS and the classifier',
  'Real satellite imagery, and a satellite basemap with vector tiles',
  'Looked-up ERG and NDMA guidance for the inferred category',
  'Alert drafts in the standard Common Alerting Protocol format for an authorized agency to review and send',
  'Smoke plume affected-zone estimate from live wind',
  'One-page PDF incident report',
];

const LIMITS = [
  'Satellites cannot see through cloud. A gap in the timeline is missing data, never zero fire.',
  'Fires smaller or cooler than the sensor threshold are missed, and one pixel is 375 m (VIIRS) or 1 km (MODIS) across, so it can hold several sources.',
  'Facility records are incomplete in parts of India. Where coverage is weak, the report says so.',
  'The data is near real time, meaning hours, not seconds. This is a triage and monitoring tool, not a dispatch system.',
  'The class is the most likely source inferred from context. It is not a confirmed cause, and low-confidence calls are flagged for a person to check.',
  'Every number on these screens is generated sample data until the live pipeline is connected.',
];

export function MethodsPage() {
  return (
    <div className="page">
      <div className="page-in">
        <PageHead title="Methods and limits" lead="What each label means, how status is decided, what this build does, and what it cannot do." />

        <div className="g2">
          <Card title="Source classes">
            <div style={{ display: 'grid', gap: 10 }}>
              {CLS.map((c) => (
                <div key={c.id}>
                  <div className="cb" style={{ fontWeight: 600 }}>
                    <Glyph cls={c.id} size={13} />
                    {c.label}
                  </div>
                  <div className="tx2 small">{c.blurb}</div>
                </div>
              ))}
            </div>
          </Card>
          <Card title="How status is decided">
            <p className="tx2" style={{ margin: '0 0 10px' }}>
              The class says what the source is. A separate rule, based on the place's own history, says whether it is behaving normally.
            </p>
            <ol className="list">
              <li>Attach the event to a site and take that site's earlier fire power readings, on a log scale.</li>
              <li>Take the median and the median absolute deviation, so a few extreme days do not skew what counts as normal.</li>
              <li>Score the event as its distance above the median in units of that spread.</li>
              <li>Call it abnormal if the score is above about 3.5 and it persists or grows, if a new source appears inside a facility, or if a quiet site switches on.</li>
              <li>With fewer than 20 clear observations, show baseline building and claim nothing.</li>
            </ol>
          </Card>
        </div>

        <div className="g2">
          <Card title="What this build does">
            <ul className="list">
              {IMPLEMENTED.map((x) => (
                <li key={x}>
                  {x} <span className="chip impl">Built, on sample data</span>
                </li>
              ))}
            </ul>
          </Card>
          <Card title="What is still proposed">
            <ul className="list">
              {PROPOSED.map((x) => (
                <li key={x}>
                  {x} <span className="chip prop">Proposed</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <Card title="Limits to keep in mind">
          <ul className="list">
            {LIMITS.map((x) => (
              <li key={x}>{x}</li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
