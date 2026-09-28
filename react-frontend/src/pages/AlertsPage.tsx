/* =====================================================================
   Alert hand-off: CAP drafts for an authorized agency to review, and a
   mock delivery log. Port of pageAlerts() in frontend/js/pages.js.
   ===================================================================== */
import { useEffect, useState } from 'react';
import { ccol } from '../lib/core';
import { TODAY, ago, fmtDT } from '../lib/time';
import { capXml } from '../data/analysis';
import { PMAP } from '../data/places';
import { winEvents } from '../state/selectors';
import { useFilters } from '../state/useFilters';
import { useFiles } from '../lib/files';
import { useToast } from '../state/ToastProvider';
import { Card, Empty, Ring } from '../components/ui/Primitives';
import { PageHead } from './PageHead';
import type { CapLogEntry } from '../lib/types';

export function AlertsPage() {
  const [f] = useFilters();
  const { copyText } = useFiles();
  const toast = useToast();
  const abn = winEvents(f)
    .filter((e) => e.status === 'abnormal')
    .sort((a, b) => b.t - a.t)
    .slice(0, 8);
  const [capId, setCapId] = useState<string | null>(null);
  const [log, setLog] = useState<CapLogEntry[]>([]);

  useEffect(() => {
    if (!capId || !abn.some((e) => e.id === capId)) setCapId(abn[0]?.id ?? null);
  }, [abn, capId]);

  const ev = abn.find((e) => e.id === capId);

  return (
    <div className="page">
      <div className="page-in">
        <PageHead title="Alert hand-off" lead="Drafts for an authorized alerting agency to review. This system does not send public alerts." />
        <div className="banner plain">
          <span>
            <span className="chip prop">Proposed</span> Only authorized government agencies can issue public alerts in India. The system
            prepares the content in the standard Common Alerting Protocol format for one of them to review. Sending here is a mock and goes
            nowhere.
          </span>
        </div>

        {ev ? (
          <div className="g21">
            <div style={{ minWidth: 0 }}>
              <Card title="Abnormal events in the current window">
                <div className="queue">
                  {abn.map((e) => {
                    const p = PMAP[e.pid];
                    return (
                      <button
                        key={e.id}
                        className="qi"
                        aria-current={e.id === capId}
                        style={{ gridTemplateColumns: 'auto minmax(0,1fr) auto' }}
                        onClick={() => setCapId(e.id)}
                      >
                        <span>
                          <Ring v={e.conf} size={36} sw={4} color={ccol(e.cls)} />
                        </span>
                        <span style={{ minWidth: 0 }}>
                          <span className="t" style={{ display: 'block' }}>
                            {p.name}
                          </span>
                          <span className="s" style={{ display: 'block' }}>
                            {p.district}, {p.state}
                          </span>
                        </span>
                        <span className="m">
                          <b className="num">{e.frp.toFixed(0)} MW</b>
                          <span>{ago(e.t)}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </Card>
            </div>

            <div style={{ display: 'grid', gap: 16, alignContent: 'start', minWidth: 0, gridTemplateColumns: 'minmax(0,1fr)' }}>
              <Card title="Alert draft" right={<span className="chip prop">Draft</span>}>
                <pre className="xml">{capXml(ev)}</pre>
                <div className="row wrap" style={{ marginTop: 12 }}>
                  <button className="btn sm" onClick={() => copyText(capXml(ev), 'Alert draft copied')}>
                    Copy draft
                  </button>
                  <button
                    className="btn sm pri"
                    onClick={() => {
                      setLog((l) => [{ t: fmtDT(TODAY), n: PMAP[ev.pid].name }, ...l]);
                      toast('Mock send logged. Nothing was transmitted.');
                    }}
                  >
                    Mock send to reviewer
                  </button>
                </div>
              </Card>
              <Card title="Mock delivery log">
                {log.length ? (
                  <div className="dist small">
                    {log.map((l, i) => (
                      <div key={i}>
                        <span>
                          {l.t} {l.n}
                        </span>
                        <span className="dim">Mock only, not sent</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="dim small">Nothing sent yet. Mock sends appear here.</div>
                )}
              </Card>
            </div>
          </div>
        ) : (
          <Empty title="No abnormal events in this window">Widen the date window on the Command page to draft one.</Empty>
        )}
      </div>
    </div>
  );
}
