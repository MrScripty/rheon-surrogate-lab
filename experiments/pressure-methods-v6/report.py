"""Current evidence and explicit remaining obligations for all five tracks."""
import html
from datetime import datetime,timezone
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha,fit_record
def build():
    evidence={}
    def keep(p):evidence[str(p.relative_to(ROOT))]=sha(p);return read(p)
    campaign=keep(DOCS/'campaign.json');sections=[];fits=[];qualifications=[];development=[]
    for arm in campaign['fits']:
        record=fit_record(arm)
        for key,device in [('qualification','cpu'),('cuda_qualification','cuda')]:
            p=ROOT/record[key]
            if p.exists():
                q=keep(p);qualifications.append(dict(arm=arm,device=device,status=q['status'],parameters=q['parameters']))
        p=ROOT/record['study']/'receipt.json'
        fits.append(dict(arm=arm,**keep(p))if p.exists()else dict(arm=arm,status='pending_or_running',generation=record['generation']))
    for p in sorted((BASE/'development').glob('*.json')):
        d=keep(p);d['parameters']=0 if not d['neural']else d['parameters'];development.append({k:d[k]for k in ['identifier','cases','qualified','parameters','median_coarse_calls','median_network_calls','worst_pressure_relative_l2','population_sha256']})
    native=[]
    for name in ['native-qualification-01.json','native-qualification-large-01.json','native-streaming-qualification-large-01.json']:
        p=BASE/name
        if p.exists():d=keep(p);native.append(dict(file=name,status=d['status'],rows=d['rows']))
    def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</table>'
    sections.append('<p>Active five-track study. No new blind results, final repeated timings, long-trajectory conclusions or deployment claims are established yet.</p>')
    sections.append('<h2>Original scope</h2><p><a href="requested-scope.txt">Exact attached request</a> · <a href="campaign.json">Requirements</a> · <a href="scene-reservation.json">Scene reservation</a> · <a href="selection-rule.json">Selection rule</a></p>')
    sections.append(table(['Track','Required evidence'],[[t['name'],'; '.join(t['required'])]for t in campaign['tracks']]))
    sections.append('<h2>Training admission</h2>'+table(['Arm','Device','Status','Parameters'],[[r['arm'],r['device'],r['status'],r['parameters']]for r in qualifications]))
    sections.append('<h2>Managed fits</h2>'+table(['Arm','Status'],[[r['arm'],r['status']]for r in fits]))
    sections.append('<p>Replay and three-iteration arms start from the same completed v5 post-coarse checkpoint and share 2,400 updates and the exact original 1,600-small-sample stream. Simpler architectures have seeded 27-parameter stencil and 9-parameter separable-filter initializations. The objective differentiates through the frozen CPU hierarchy and GCR, including device copies; actual forward and coarse work must be reported separately from update count. All use the original native training bank; v6 reserved scenes never enter fitting.</p>')
    sections.append('<h2>Consumed-data development</h2>'+table(['Policy','Qualified','Cases','Parameters','Median hierarchy cycles','Median neural calls','Worst pressure rel. L2'],[[r[k]for k in ['identifier','qualified','cases','parameters','median_coarse_calls','median_network_calls','worst_pressure_relative_l2']]for r in development]))
    sections.append('<p>All 254 requests are previously consumed inputs. This screen is development evidence. Its exploratory times may overlap training and cannot establish a final speed gain. The first exploratory controller failed at output serialization because the result directory did not exist; it produced no admitted result and is preserved. Fresh execution 02 creates the directory first and rejects source changes during execution.</p>')
    sections.append('<h2>Native inference correctness</h2><p>The original native kernel and the streaming-plane implementation preserve replicated boundaries at every convolution stage. Parity checks include degenerate axes, odd rectangles, full 128³ and full 256³. They verify finite, cell-zero-gauged float64 results and output ownership. Floating-point accumulation differs from PyTorch; full returned-field solver gates remain necessary. These one-shot profiles are correctness diagnostics, not isolated repeated timing.</p>'+table(['Check','Status','Full-grid shapes'],[[r['file'],r['status'],[x['counts_zyx']for x in r['rows']]]for r in native]))
    sections.append('<h2>Execution amendment</h2><p>The first three-iteration attempt reached its admitted1800-second ceiling and failed/drained; its partial work is retained and excluded from completed fits. Generation02 preserves the scientific configuration, initialization, samples and1800-second cap. It replaces generic CPU differentiation through the frozen hierarchy with an independently verified linear adjoint, including the cell-zero gauge transpose. No numerical gradient is dropped.</p><h2>Remaining work</h2><p>Finish four fits and fresh CPU restores; evaluate trained and selective-use variants; calibrate adaptive reuse from development costs; freeze selection; capture untouched blind and paired larger-grid scenes; run two 256-step independent trajectory comparisons; perform isolated repeated total-policy/native timing; generate figures and complete a requirement-by-requirement evidence audit. Application drift limits remain diagnostic, and no model promotion is requested.</p>')
    payload=dict(status='active',updated_utc=datetime.now(timezone.utc).isoformat(),fits=fits,qualifications=qualifications,development=development,native_qualifications=native,evidence=evidence,source_sha256=sha(__file__),promotion=False)
    write(DOCS/'results.json',payload)
    (DOCS/'results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Pressure methods v6</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:24px}p{line-height:1.5}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}</style><h1>Pressure methods v6: cost-aware local correction</h1>'+''.join(sections)+'</html>')
    print('Rendered current five-track evidence with incomplete obligations explicit',flush=True)
if __name__=='__main__':build()
