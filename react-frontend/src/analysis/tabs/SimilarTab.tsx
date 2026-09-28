/* =====================================================================
   Similar events, and the analyst review tool that seeds a labelled set.
   Port of tabSimilar() and tabReview() in frontend/js/analysis/tabs.js.
   ===================================================================== */
import { useState } from 'react';
import { ccol } from '../../lib/core';
import { fmtD } from '../../lib/time';
import { similarOf } from '../../data/analysis';
import { CLS, CLSMAP } from '../../data/classes';
import { ClassBadge } from '../../components/ui/Glyph';
import { Icon } from '../../components/ui/Icon';
import { Empty, Ring } from '../../components/ui/Primitives';
import { Sec } from '../parts';
import { useReviews, reviewsCsv } from '../../state/ReviewsProvider';
import { useFiles } from '../../lib/files';
import { useToast } from '../../state/ToastProvider';
import type { ClassId, Event, Verdict } from '../../lib/types';

export function SimilarTab({ ev, onGo }: { ev: Event; onGo: (placeId: string) => void }) {
  const list = similarOf(ev);
  return (
    <>
      <div className="banner plain small" style={{ marginBottom: 12 }}>
        <span>
          <b>Sample list.</b> The live build finds these by comparing the full feature vector against past events with a known outcome.
        </span>
      </div>
      {list.length ? (
        <div className="queue">
          {list.map((x) => (
            <button
              key={x.e.id}
              className="qi"
              style={{ gridTemplateColumns: 'auto minmax(0,1fr) auto' }}
              onClick={() => onGo(x.p.id)}
            >
              <span>
                <Ring v={x.sim} size={40} sw={4} color={ccol(x.e.cls)} />
              </span>
              <span style={{ minWidth: 0 }}>
                <span className="t" style={{ display: 'block' }}>
                  {x.p.name}
                </span>
                <span className="s" style={{ display: 'block' }}>
                  {fmtD(x.e.t)}, {x.p.district}, {x.p.state}
                </span>
                <span className="chips">
                  <ClassBadge cls={x.e.cls} />
                  <span className={'chip ' + (x.outcome.startsWith('Marked') ? 'rev' : '')}>{x.outcome}</span>
                </span>
              </span>
              <span className="m">
                <b className="num">{x.e.frp.toFixed(0)} MW</b>
                <span>{Math.round(x.sim * 100)}% alike</span>
              </span>
            </button>
          ))}
        </div>
      ) : (
        <Empty title="Nothing earlier to compare with">This is the first event of its kind in the sample.</Empty>
      )}
    </>
  );
}

const VERDICTS: [Verdict, string][] = [
  ['confirm', 'Confirm the model’s class'],
  ['change', 'Change the class'],
  ['false', 'This is a false alarm'],
  ['field', 'Needs a field check'],
];

export function ReviewTab({ ev }: { ev: Event }) {
  const { reviews, addReview, total } = useReviews();
  const { saveFile } = useFiles();
  const toast = useToast();
  const [verdict, setVerdict] = useState<Verdict>('confirm');
  const [cls, setCls] = useState<ClassId>(ev.cls2);
  const [note, setNote] = useState('');
  const list = reviews[ev.id] || [];

  const save = () => {
    addReview(ev.id, verdict, verdict === 'change' ? cls : null, note.trim());
    toast('Review saved on this device');
    setNote('');
  };

  return (
    <>
      <p className="sum" style={{ margin: '0 0 12px' }}>
        The model said <b>{CLSMAP[ev.cls].label}</b>, {Math.round(ev.conf * 100)}% sure
        {ev.review ? ', and flagged it for a person to check' : ''}. Your call is saved on this device. Confirmed and corrected calls can be
        exported and used as real training labels later.
      </p>
      <section className="card">
        <div className="cbody">
          <fieldset className="rv" style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="sr">Your verdict</legend>
            {VERDICTS.map(([v, l]) => (
              <label className="rvopt" key={v}>
                <input type="radio" name="rv" value={v} checked={verdict === v} onChange={() => setVerdict(v)} />
                <span>{l}</span>
              </label>
            ))}
          </fieldset>
          {verdict === 'change' && (
            <div style={{ margin: '8px 0' }}>
              <label className="dim small" htmlFor="rv-sel">
                Should be
              </label>
              <br />
              <select id="rv-sel" className="sel" value={cls} onChange={(e) => setCls(e.target.value as ClassId)}>
                {CLS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <label className="dim small" htmlFor="rv-note" style={{ display: 'block', marginTop: 10 }}>
            Note
          </label>
          <textarea
            id="rv-note"
            rows={4}
            placeholder="What did you see? For example: the satellite view shows a new pad next to the stack."
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <div className="row wrap" style={{ marginTop: 10 }}>
            <button className="btn pri" onClick={save}>
              Save my review
            </button>
          </div>
        </div>
      </section>

      {list.length > 0 && (
        <Sec title="Earlier reviews of this event">
          <div className="dist small">
            {[...list].reverse().map((r, i) => (
              <div key={i}>
                <span>
                  <b>{VERDICTS.find((x) => x[0] === r.verdict)?.[1]}</b>
                  {r.cls ? ', as ' + CLSMAP[r.cls].label : ''}
                  {r.note && (
                    <>
                      <br />
                      <span className="dim">{r.note}</span>
                    </>
                  )}
                </span>
                <span className="dim">{r.at}</span>
              </div>
            ))}
          </div>
        </Sec>
      )}

      <div className="row between" style={{ marginTop: 14 }}>
        <span className="dim small">
          {total} review{total === 1 ? '' : 's'} saved on this device
        </span>
        <button className="btn sm" disabled={!total} onClick={() => saveFile('agni-netra-analyst-reviews.csv', reviewsCsv(reviews), 'CSV')}>
          <Icon name="download" size={14} />
          Export all as CSV
        </button>
      </div>
    </>
  );
}
