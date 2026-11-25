"""Read-only CPU stage observer. Never signals or writes to the trainer/source."""
import os,sys,json,time,csv,hashlib,re,math,subprocess,argparse,fcntl
from pathlib import Path
from datetime import datetime,timezone
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='offline')
sys.dont_write_bytecode=True
SOURCE=Path('/root/autodl-tmp/GLIM-main-A3-v2')
RUN=Path('/root/autodl-tmp/A3-v2-V3-50-A800/run-02')
PILOT=SOURCE/'runs/A3-v2-pilot-5-r2'
OUT=Path('/root/autodl-tmp/A3-v2-V3-50-A800-stage-reports')
CLASSES=['negative','neutral','positive']

def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(p.read_text())
def stamp(p):
    s=p.stat(); return (s.st_size,s.st_mtime_ns,s.st_ino)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def jsonwrite(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)); os.replace(tmp,p)
def jsonlines(p):
    result=[]
    if p.exists():
        for line in p.read_text().splitlines():
            try: result.append(json.loads(line))
            except json.JSONDecodeError: pass
    return result

def checkpoint_info(p,epoch):
    import torch
    before=stamp(p)
    c=torch.load(p,map_location='cpu',weights_only=False)
    assert c['epoch']==epoch and c['state_dict'] and c['optimizer_states'] and c['loops']
    progress=c['loops']['fit_loop']['epoch_progress']
    # Lightning saves at validation end before incrementing epoch 'completed'.
    # 'processed' counts the finished epoch at this checkpoint boundary.
    assert progress['total']['processed']>=epoch+1,progress
    result=dict(path=str(p),sha256=sha(p),epoch=c['epoch'],global_step=c['global_step'],
        bytes=before[0],model_parameter_tensors=len(c['state_dict']),
        adam_parameter_states=len(c['optimizer_states'][0]['state']),epoch_progress=progress,
        validation='CPU deserialization, completed-loop check, SHA256, unchanged file stat')
    assert before==stamp(p),'Checkpoint changed during inspection'
    del c
    return result

def history(root):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    rows=[]; audits=[]
    for p in root.glob('wandb/offline-run-*/*.wandb'):
        ds=DataStore()
        # Library open_for_scan uses r+b; explicitly implement rb-only scanning.
        ds._fname=str(p); ds._fp=p.open('rb'); ds._index=0
        ds._size_bytes=p.stat().st_size; ds._opened_for_scan=True
        tail_error=None; count=0
        try:
            ds._read_header()
            while True:
                raw=ds.scan_data()
                if raw is None: break
                rec=wandb_internal_pb2.Record(); rec.ParseFromString(raw)
                if rec.HasField('history'):
                    row={}
                    for item in rec.history.item:
                        key=item.key or '/'.join(item.nested_key)
                        row[key]=json.loads(item.value_json)
                    rows.append(row); count+=1
        except Exception as e: tail_error=repr(e)
        finally: ds._fp.close()
        audits.append(dict(path=str(p),history_records=count,unread_tail_or_error=tail_error))
    return rows,audits

def collect_series(target):
    hist=[]; audits=[]
    for root in [PILOT,RUN]:
        r,a=history(root); hist.extend(r); audits.extend(a)
    series=[]
    for ep in range(target+1):
        root=PILOT if ep<=4 else RUN
        p=root/f'epoch-{ep:02d}-metrics.json'
        if not p.exists():
            series.append(dict(epoch=ep,status='未记录')); continue
        m=read_json(p)
        row=dict(epoch=ep,status='已记录',metrics_file=str(p),metrics=m)
        sampled=[r.get('train/loss_lm') for r in hist if r.get('epoch')==ep and isinstance(r.get('train/loss_lm'),(int,float))]
        row['train_lm_logged_point_mean']=sum(sampled)/len(sampled) if sampled else None
        row['train_lm_logged_points']=len(sampled)
        row['train_lm_note']='已记录训练点算术平均，不是全 batch 加权 epoch 均值' if sampled else '未记录可读取的训练历史；不以末 batch 冒充平均'
        series.append(row)
    return series,audits

def view(row):
    m=row.get('metrics',{})
    hist=[m.get(f'val/code_usage/{i}') for i in range(128)]
    valid=all(isinstance(x,(int,float)) for x in hist) and sum(hist)>0
    return dict(epoch=row['epoch'],train_lm_logged_point_mean=row.get('train_lm_logged_point_mean'),
        train_lm_logged_points=row.get('train_lm_logged_points',0),val_lm=m.get('val/loss_lm'),
        sentiment_loss=m.get('val/loss_disc_emotion'),macro_f1=m.get('val/disc_macro_f1'),
        balanced_accuracy=m.get('val/disc_balanced_accuracy'),vq_loss=m.get('val/loss_vq'),
        active_codes=sum(x>0 for x in hist) if valid else None,
        perplexity=m.get('val/codebook_perplexity'),top1_ratio=max(hist)/sum(hist) if valid else None)

def curves(folder,series):
    # Standalone vector charts from observed values only; gaps remain gaps.
    rows=[view(r) for r in series]
    panels=[('LM Loss',[('train_lm_logged_point_mean','#2368a2'),('val_lm','#d65f24')]),
            ('Sentiment Macro-F1',[('macro_f1','#2368a2')]),('Codebook Perplexity',[('perplexity','#2368a2')])]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780"><rect width="1000" height="780" fill="white"/>']
    for n,(title,keys) in enumerate(panels):
        top=30+n*250; vals=[r.get(k) for r in rows for k,_ in keys if isinstance(r.get(k),(int,float)) and math.isfinite(r[k])]
        if not vals: continue
        low,high=min(vals),max(vals); span=max(high-low,1e-6)
        svg.append(f'<text x="65" y="{top}" font-size="17">{title} (recorded epochs only)</text>')
        svg.append(f'<path d="M65 {top+15}V{top+190}H950" fill="none" stroke="black"/>')
        svg.append(f'<text x="5" y="{top+25}">{high:.3g}</text><text x="5" y="{top+185}">{low:.3g}</text>')
        for ki,(key,color) in enumerate(keys):
            svg.append(f'<text x="{65+ki*450}" y="{top+225}" fill="{color}" font-size="12">{key}</text>')
            previous=None
            for r in rows:
                v=r.get(key)
                if not isinstance(v,(int,float)) or not math.isfinite(v): previous=None; continue
                x=65+885*r['epoch']/max(rows[-1]['epoch'],1); y=top+185-165*(v-low)/span
                if previous:
                    svg.append(f'<path d="M{previous[0]:.2f} {previous[1]:.2f}L{x:.2f} {y:.2f}" stroke="{color}" fill="none"/>')
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>'); previous=(x,y)
        svg.append(f'<text x="65" y="{top+207}">epoch 0</text><text x="880" y="{top+207}">epoch {rows[-1]["epoch"]}</text>')
    svg.append('</svg>'); (folder/'training_curves.svg').write_text(''.join(svg))

def report(epoch,checkpoint,folder):
    info=checkpoint_info(checkpoint,epoch)
    series,audits=collect_series(epoch)
    assert series[-1].get('metrics'), 'Waiting for completed validation metrics'
    current=view(series[-1]); m=series[-1]['metrics']; baseline=view(series[4])
    diag=[r for r in jsonlines((PILOT if epoch<=4 else RUN)/'diagnostics.jsonl') if r.get('epoch')==epoch and r.get('step',0)<=info['global_step']]
    last=diag[-1] if diag else None
    raw=(RUN/'logs/train.log').read_text(errors='replace')
    speeds=re.findall(r'Epoch '+str(epoch)+r':[^\r\n]*?([0-9.]+)it/s',raw)
    anomalies=[line[:500] for line in raw.replace('\r','\n').splitlines() if re.search(r'CUDA out of memory|\bOOM\b|\bNaN\b|Traceback|RuntimeError',line,re.I)]
    class_metrics={c:{k:m.get(f'val/disc/{c}/{k}') for k in ['precision','recall','f1']} for c in CLASSES}
    confusion=[[m.get(f'val/disc/confusion/{a}/{b}') for b in CLASSES] for a in CLASSES]
    recent=[view(r) for r in series[max(0,epoch-4):] if r.get('metrics')]
    findings=[]
    if len(recent)>=2:
        a,b=recent[0],recent[-1]
        if all(isinstance(x,(int,float)) for x in [a['val_lm'],b['val_lm'],a['train_lm_logged_point_mean'],b['train_lm_logged_point_mean']]):
            findings.append('最近五个已记录 epoch 的训练 LM 采样均值下降、验证 LM 上升，存在过拟合迹象（非因果结论）。' if b['val_lm']>a['val_lm'] and b['train_lm_logged_point_mean']<a['train_lm_logged_point_mean'] else '最近五个已记录 epoch 未同时出现训练 LM 下降与验证 LM 上升；这不能排除过拟合。')
        else: findings.append('训练或验证 LM 趋势数据不足，过拟合判断受限。')
    for key,label in [('perplexity','码本 Perplexity'),('macro_f1','情感 Macro-F1'),('balanced_accuracy','Balanced Accuracy')]:
        x,y=current.get(key),baseline.get(key)
        findings.append(f'{label}：epoch 4={y}，epoch {epoch}={x}；'+('较 Pilot 下降，需关注退化。' if isinstance(x,(int,float)) and isinstance(y,(int,float)) and x<y else '未观察到较 Pilot 下降或数据不足。'))
    findings.append('高码本利用率不等于情感语义有效；不凭总体 Projection Std 断言跨样本可分性。')
    best={}
    if epoch==49:
        for key,name,reverse in [('macro_f1','best_sentiment_macro_f1',True),('val_lm','best_validation_lm',False)]:
            available=[view(r) for r in series if isinstance(view(r).get(key),(int,float)) and math.isfinite(view(r)[key])]
            chosen=sorted(available,key=lambda x:x[key],reverse=reverse)[0] if available else None
            if chosen:
                root=PILOT if chosen['epoch']<=4 else RUN
                paths=list((root/'checkpoints').glob(f'epoch={chosen["epoch"]:02d}-step=*.ckpt'))
                best[name]=dict(metrics=chosen,checkpoint=checkpoint_info(paths[0],chosen['epoch']) if len(paths)==1 else None)
        best_f1=best.get('best_sentiment_macro_f1',{}).get('metrics',{})
        at29=view(series[29])
        if isinstance(current.get('macro_f1'),(int,float)) and best_f1.get('macro_f1',0)>current['macro_f1']+.01:
            findings.append(f'继续训练建议：不建议直接延长。最佳情感验证轮为 epoch {best_f1["epoch"]}，最终 Macro-F1 比最佳低超过 0.01，应先诊断退化并保留最佳轮。差值阈值是诊断规则，不是显著性检验。')
        elif all(isinstance(x,(int,float)) for x in [current.get('macro_f1'),at29.get('macro_f1'),current.get('val_lm'),at29.get('val_lm')]) and current['macro_f1']>at29['macro_f1'] and current['val_lm']<=at29['val_lm']:
            findings.append('继续训练建议：相比 epoch 29，最终情感 Macro-F1 上升且验证 LM 未升高，可考虑经人工确认的有限续训对照；单次运行不足以保证继续获益。本程序不会启动训练。')
        else:
            findings.append('继续训练建议：现有指标没有同时支持情感与 LM 持续改善，不建议盲目延长；先分析码本语义、类别混淆与最佳轮。不会自动续训。')
    telemetry=[r for r in jsonlines(OUT/'telemetry.jsonl') if r.get('progress',{}).get('epoch')==epoch]
    gpu_samples=[]
    for r in telemetry:
        try:
            parts=r['gpu_csv'].splitlines()[0].split(',')
            gpu_samples.append((float(parts[-3]),float(parts[-1])))
        except (ValueError,IndexError,KeyError): pass
    gpu_summary=dict(sample_count=len(gpu_samples),memory_mib_min=min((r[0] for r in gpu_samples),default=None),
        memory_mib_max=max((r[0] for r in gpu_samples),default=None),
        utilization_percent_min=min((r[1] for r in gpu_samples),default=None),
        utilization_percent_max=max((r[1] for r in gpu_samples),default=None))
    try: latest_progress=read_json(RUN/'progress.json')
    except Exception: latest_progress=None
    result=dict(kind='READ_ONLY_STAGE_REPORT',generated_at_utc=now(),checkpoint=info,metrics=current,
        baseline_epoch4=baseline,comparison_epoch29=view(series[29]) if epoch==49 else None,
        class_order=CLASSES,per_class=class_metrics,confusion_matrix=confusion,
        confusion_axes='rows=true, columns=predicted',projection_mean=None,projection_mean_status='未记录',
        projection_std=last.get('projection_std') if last else None,projection_observation_step=last.get('step') if last else None,
        projection_note='最后一个已记录训练 batch 的有效 token 总体 Std；不是整轮均值',
        train_sentiment_loss_last_batch=m.get('train/loss_disc_emotion'),
        train_vq_loss_last_batch=m.get('train/loss_vq'),last_training_diagnostic=last,
        gpu_telemetry=telemetry,gpu_summary=gpu_summary,latest_training_progress=latest_progress,
        training_pid_exists=Path('/proc/2183/cmdline').exists(),
        gpu_telemetry_note='独立监控启动后每60秒采样；非完整历史峰值',
        logged_training_speed_it_s=float(speeds[-1]) if speeds else None,
        anomaly_log_matches=anomalies,anomaly_scope='日志前缀截至报告读取时；无匹配不等于证明从未异常',
        source_metrics=m,history_audit=audits,series=series,best_checkpoints=best,findings=findings,
        missing=['Projection Mean','精确全 batch 训练 LM epoch 均值','监控启动前 GPU 时间序列'],
        metric_notes='Active Codes/Top-1 直接汇总已记录的验证码本直方图；无补齐、无模型推理。')
    jsonwrite(folder/'report.json',result)
    rows=[view(r) for r in series]
    with (folder/'epochs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (folder/'per_class.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['class','precision','recall','f1'])
        for c in CLASSES: w.writerow([c,*[class_metrics[c][k] for k in ['precision','recall','f1']]])
    with (folder/'confusion_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['true/predicted',*CLASSES])
        for c,row in zip(CLASSES,confusion): w.writerow([c,*row])
    def fmt(v): return '未记录' if v is None else f'{v:.6g}' if isinstance(v,float) else str(v)
    lines=[f'# A3-v2 V3-50：Epoch {epoch} 中文阶段报告','',f'Checkpoint：`{info["path"]}`',
        f'Global step：{info["global_step"]}；SHA256：`{info["sha256"]}`','',
        '## 实际验证指标与 Pilot 对比','','| 指标 | Epoch 4 | 当前轮 |','|---|---:|---:|']
    for k in current:
        if k!='epoch': lines.append(f'| {k} | {fmt(baseline.get(k))} | {fmt(current[k])} |')
    lines+=['','训练 LM 为 W&B 已记录训练点的算术平均，不是全 batch epoch 均值。详见 epochs.csv。',
        '','## 类别指标','','| 类别 | Precision | Recall | F1 |','|---|---:|---:|---:|']
    for c in CLASSES: lines.append('| '+c+' | '+' | '.join(fmt(class_metrics[c][k]) for k in ['precision','recall','f1'])+' |')
    lines+=['','混淆矩阵（行是真实类别，列是预测类别）：','','| true/pred | negative | neutral | positive |','|---|---:|---:|---:|']
    for c,row in zip(CLASSES,confusion): lines.append('| '+c+' | '+' | '.join(map(fmt,row))+' |')
    lines+=['',f'Projection Mean：未记录；Projection Std：{fmt(result["projection_std"])}，观察 step={result["projection_observation_step"]}。',
        f'日志记录训练速度：{fmt(result["logged_training_speed_it_s"])} it/s；异常关键字匹配 {len(anomalies)} 条，原文保存在 JSON。',
        f'当前轮 GPU 采样记录 {len(telemetry)} 条，详见 report.json；缺失历史不填补。','', '## 趋势与判断','']+findings
    lines+=['',f'GPU 显存采样范围：{fmt(gpu_summary["memory_mib_min"])}–{fmt(gpu_summary["memory_mib_max"])} MiB；利用率采样范围：{fmt(gpu_summary["utilization_percent_min"])}–{fmt(gpu_summary["utilization_percent_max"])}%。',
        f'报告生成时最新训练进度（来自 progress.json，可能已进入下一轮）：{latest_progress}']
    lines+=['','## 训练曲线','','![实际记录曲线](training_curves.svg)','',
        '每个观测点来自实际日志；缺失轮次保留缺口。Epoch 49 报告覆盖 0–49，未记录项仍为空。']
    if best:
        lines+=['','## 最佳验证 checkpoint 与最终 checkpoint','',
            '分别按验证 Macro-F1 最大、验证 LM Loss 最小选取，避免用总 Loss 选优。','']
        for name,v in best.items(): lines.append(f'- {name}：{v["metrics"]}；checkpoint：{v["checkpoint"]}')
        lines+=['',f'Epoch 29 对照：{result["comparison_epoch29"]}',f'最终 Epoch 49：{current}']
    lines+=['','## 缺失项及后续限制','']+['- '+s for s in result['missing']]+[
        '- 本程序不更改训练、不发信号、不删除 checkpoint、不启动额外训练。',
        '- 保留原始日志；只在独立阶段报告目录写 JSON、CSV、Markdown、SVG。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n'); curves(folder,series)
    return info

def sample_gpu():
    row=dict(time_utc=now())
    try: row['progress']=read_json(RUN/'progress.json')
    except Exception: row['progress']={}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        row['gpu_csv']=p.stdout.strip(); row['error']=p.stderr.strip()
    except Exception as e: row['error']=repr(e)
    with (OUT/'telemetry.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--test-pilot',action='store_true'); args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.test_pilot:
        dest=OUT/'self-test-epoch04'; dest.mkdir(exist_ok=True)
        report(4,PILOT/'checkpoints/epoch=04-step=9940.ckpt',dest)
        print('PASS actual pilot checkpoint and report generation',flush=True); return
    lock=(OUT/'monitor.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'monitor.pid').write_text(str(os.getpid()))
    seen={}
    while True:
        sample_gpu()
        for epoch in [29,49]:
            final=OUT/f'epoch-{epoch:02d}'
            if (final/'READY.json').exists(): continue
            paths=list((RUN/'checkpoints').glob(f'epoch={epoch:02d}-step=*.ckpt'))
            if len(paths)!=1: continue
            path=paths[0]; s=stamp(path)
            if seen.get(epoch)!=s: seen[epoch]=s; continue
            # Two observations >=60 seconds apart, completed loop, successful
            # full CPU load, hash, and unchanged stat before/after reading.
            tmp=OUT/f'epoch-{epoch:02d}.building'; tmp.mkdir(exist_ok=True)
            try:
                info=report(epoch,path,tmp)
                assert s==stamp(path)
                jsonwrite(tmp/'READY.json',dict(status='COMPLETE',time_utc=now(),checkpoint=info))
                tmp.rename(final)
                print('REPORT_READY',str(final),flush=True)
            except Exception as e:
                jsonwrite(OUT/f'epoch-{epoch:02d}-retry.json',dict(time_utc=now(),error=repr(e)))
                print('WAIT_RETRY',epoch,repr(e),flush=True)
        done=all((OUT/f'epoch-{ep:02d}'/'READY.json').exists() for ep in [29,49])
        jsonwrite(OUT/'status.json',dict(pid=os.getpid(),updated_at_utc=now(),complete=done,
            nodes={str(ep):(OUT/f'epoch-{ep:02d}'/'READY.json').exists() for ep in [29,49]}))
        if done: break
        time.sleep(60)

if __name__=='__main__': main()
