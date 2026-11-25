"""Bounded five-epoch pilot, with non-mutating monitoring and fail-closed guards."""
import json
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from deployment.local_logging import LocalWandbLogger


class Guard(Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.hist = torch.zeros(128, dtype=torch.long)
        self.stds = []
        self.updated = torch.zeros(128, dtype=torch.bool)
        self.grad = {}

    def on_fit_start(self, trainer, model):
        def projected(module, args, out):
            if trainer.training:
                self.std = float(out.detach().float().std())
        def quantized(module, args, out):
            if trainer.training:
                self.hist += out['code_usage_histogram'].detach().cpu()
        self.handles = [model.vq.projection.register_forward_hook(projected),
                        model.vq.register_forward_hook(quantized)]
        self.previous = model.vq.codebook.weight.detach().clone()

    def on_before_optimizer_step(self, trainer, model, optimizer):
        for name, module in [('encoder', model.eeg_encoder), ('projection', model.vq.projection), ('codebook', model.vq.codebook)]:
            grads = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
            value = torch.stack([g.square().sum() for g in grads]).sum().sqrt() if grads else torch.tensor(float('nan'))
            self.grad[name] = float(value)
            if not torch.isfinite(value) or value > 10000 or value == 0:
                raise RuntimeError(f'Gradient guard: {name}={value}')

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if not torch.isfinite(outputs['loss']).all():
            raise RuntimeError('Nonfinite training loss')
        current = model.vq.codebook.weight.detach()
        self.updated |= current.ne(self.previous).any(1).cpu()
        self.previous.copy_(current)
        step = trainer.global_step
        if step == 1 or step % 100 == 0:
            p = self.hist.float()/self.hist.sum().clamp_min(1)
            row = dict(step=step, epoch=trainer.current_epoch, active_codes=int(self.hist.gt(0).sum()),
                       perplexity=float((-(p*p.clamp_min(1e-30).log()).sum()).exp()),
                       top1_ratio=float(p.max()), projection_std=self.std,
                       codebook_update_count=int(self.updated.sum()), gradient_norm=self.grad,
                       histogram=self.hist.tolist(), loss=float(outputs['loss']),
                       gate=float(model.attribute.gate.sigmoid()) if hasattr(model.attribute, 'gate') else None)
            with (self.output/'diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print('VQ_PILOT', json.dumps({k:v for k,v in row.items() if k!='histogram'}), flush=True)
            self.stds.append(self.std)
            self.hist.zero_(); self.updated.zero_()
            if step >= 100 and row['active_codes'] <= 1:
                raise RuntimeError('Single-code collapse over monitoring window')
            if len(self.stds)>=4 and self.std < .25*self.stds[0] and all(a>b for a,b in zip(self.stds[-4:],self.stds[-3:])):
                raise RuntimeError('Persistent projection variance contraction')

    def on_validation_end(self, trainer, model):
        metrics = {k:float(v) for k,v in trainer.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1}
        (self.output/f'epoch-{trainer.current_epoch:02d}-metrics.json').write_text(json.dumps(metrics,indent=2))
        # Preserve generated text files before the next validation overwrites them.
        import shutil
        dest=self.output/f'epoch-{trainer.current_epoch:02d}-samples'; dest.mkdir(exist_ok=True)
        if trainer.current_epoch==0 or (trainer.current_epoch+1)%model.full_val_interval==0:
            for p in self.output.glob('*.csv'):
                shutil.copy2(p,dest/p.name)

    def on_exception(self, trainer, model, exception):
        trainer.save_checkpoint(self.output/'checkpoints'/'diagnostic-stop.ckpt')
        (self.output/'STOP.json').write_text(json.dumps(dict(reason=str(exception),step=trainer.global_step,epoch=trainer.current_epoch),indent=2))


def train_pilot(model, dm, initialization, report, output='runs/A3-v2-pilot-5'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir()
    saved=torch.load(initialization,map_location='cpu',weights_only=False)
    missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
    assert not unexpected and all(k.startswith('text_model.') for k in missing)
    model.vq.distance=saved['hyper_parameters']['vq_distance']
    model.hparams.vq_distance=model.vq.distance
    model.train()
    (output/'selection.json').write_text(json.dumps(report,indent=2))
    L.seed_everything(42,workers=True)
    # Reuse the already audited data in RAM; never reload the 14 GB pickle.
    dm.setup=lambda stage: None
    logger=LocalWandbLogger(project='glim',group='A3-v2',save_dir=str(output),offline=True)
    trainer=L.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=5,
        use_distributed_sampler=False,num_sanity_val_steps=0,logger=logger,
        callbacks=[Guard(output),ModelCheckpoint(dirpath=output/'checkpoints',save_top_k=-1,
            every_n_epochs=1,save_last=True,filename='epoch={epoch:02d}-step={step}',auto_insert_metric_name=False)])
    trainer.fit(model,datamodule=dm)
    (output/'COMPLETE.json').write_text(json.dumps(dict(epochs=5,step=trainer.global_step,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30),indent=2))
    config=dict(model.hparams); config.update(variant=model.variant)
    (output/'proposed-50-epoch-model-config.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':
    import os
    from model.glim_multitask import GLIMMultitask
    from data.datamodule import GLIMDataModule
    base=Path('runs/A3-v2-initialization')
    report=json.loads((base/'report.json').read_text())
    assert all(v['real_batch_smoke']['status']=='PASS' for v in report['variants'].values())
    init=base/report['selection']['variant']/'initialization.ckpt'
    os.environ['GLIM_TEXT_MODEL_PATH']=str(Path('models/flan-t5-large').resolve())
    model=GLIMMultitask.from_glim_checkpoint(init)
    dm=GLIMDataModule('data/tmp/zuco_eeg_label_8variants.df',bsz_train=model.bsz_train,bsz_val=model.bsz_val,num_workers=4)
    dm.label_reference=Path('data/tmp/zuco_label_8variants.df')
    dm.manifest_path=base/'pilot_split_manifest.csv'
    dm.setup('fit')
    assert (base/'split_manifest.csv').read_bytes()==dm.manifest_path.read_bytes()
    train_pilot(model,dm,init,report,output='runs/A3-v2-pilot-5-r2')
