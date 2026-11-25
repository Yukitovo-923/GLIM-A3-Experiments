"""Resume the completed V3 pilot without writing to its source tree."""
import os,sys,json,hashlib
from pathlib import Path

SOURCE=Path('/root/autodl-tmp/GLIM-main-A3-v2')
OUTPUT=Path('/root/autodl-tmp/A3-v2-V3-50-A800')
PILOT=SOURCE/'runs/A3-v2-pilot-5-r2'
CHECKPOINT=PILOT/'checkpoints/epoch=04-step=9940.ckpt'
os.chdir(OUTPUT)
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE))
for key,relative in [('XDG_CACHE_HOME','cache'),('HF_HOME','cache/hf'),('WANDB_DIR','wandb'),
                     ('WANDB_CACHE_DIR','cache/wandb'),('WANDB_CONFIG_DIR','config/wandb'),
                     ('WANDB_DATA_DIR','wandb-data'),('TMPDIR','tmp'),('CUDA_CACHE_PATH','cache/cuda'),
                     ('MPLCONFIGDIR','cache/matplotlib'),('TORCH_HOME','cache/torch')]:
    p=OUTPUT/relative; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
os.environ.update(GLIM_TEXT_MODEL_PATH=str(SOURCE/'models/flan-t5-large'),
                  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',
                  PYTHONDONTWRITEBYTECODE='1')

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback,ModelCheckpoint
from model.glim_multitask import GLIMMultitask
from data.datamodule import GLIMDataModule
from deployment.local_logging import LocalWandbLogger
from vq_pilot import Guard

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def equal_tree(a,b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b,(a,b)

class VerifyRestore(Callback):
    def __init__(self,saved): self.saved=saved
    def on_train_start(self,trainer,model):
        assert trainer.global_step==9940,trainer.global_step
        current=model.state_dict()
        for k,v in self.saved['state_dict'].items(): equal_tree(v,current[k])
        equal_tree(self.saved['optimizer_states'][0],trainer.optimizers[0].state_dict())
        assert (OUTPUT/'split_manifest.csv').read_bytes()==(PILOT/'split_manifest.csv').read_bytes()
        self.report=dict(status='PASS',model_parameters_exact=True,adam_state_exact=True,
            optimizer_parameter_states=len(trainer.optimizers[0].state),global_step=trainer.global_step,
            loop_state_restored=True,max_epochs=trainer.max_epochs,precision=str(trainer.precision),
            checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),split_manifest_identical=True)
        (OUTPUT/'restore_validation.json').write_text(json.dumps(self.report,indent=2))
        print('RESTORE_VALIDATED',json.dumps(self.report),flush=True)
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        if not hasattr(self,'first_batch'):
            assert trainer.current_epoch==5 and trainer.global_step==9940,(trainer.current_epoch,trainer.global_step)
            self.first_batch=True
            self.report.update(first_epoch=trainer.current_epoch,first_batch_idx=batch_idx)
            (OUTPUT/'restore_validation.json').write_text(json.dumps(self.report,indent=2))
            print('FIRST_RESUMED_BATCH',json.dumps(self.report),flush=True)
            self.saved=None
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        if trainer.global_step<=9943 or trainer.global_step%100==0:
            assert torch.isfinite(outputs['loss']).all()
            row=dict(epoch=trainer.current_epoch,batch=batch_idx,global_step=trainer.global_step,
                     loss=float(outputs['loss']),cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30)
            (OUTPUT/'progress.json').write_text(json.dumps(row,indent=2))
            print('RESUME_PROGRESS',json.dumps(row),flush=True)

def main():
    if (OUTPUT/'run_started.json').exists(): raise RuntimeError('Refusing duplicate launch')
    assert digest(CHECKPOINT)=='7ad2a5ee5d85bf06df1d8e33d38aa714f38059ef0a083110e2e1560941cf6d7d'
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert saved['epoch']==4 and saved['global_step']==9940 and saved['loops'] and saved['optimizer_states']
    h=saved['hyper_parameters']
    assert h['vq_distance']=='cosine' and h['bsz_train']==72 and h['codebook_size']==128 and h['code_dim']==256
    L.seed_everything(42,workers=True)
    model=GLIMMultitask.from_glim_checkpoint(CHECKPOINT)
    # All non-T5 parameters were validated strictly above. GLIM omits frozen T5
    # weights from checkpoints and reloads them from the existing local model.
    model.strict_loading=False
    model.train()
    dm=GLIMDataModule(str(SOURCE/'data/tmp/zuco_eeg_label_8variants.df'),
        bsz_train=model.bsz_train,bsz_val=model.bsz_val,bsz_test=model.bsz_retrieval,num_workers=4)
    dm.label_reference=SOURCE/'data/tmp/zuco_label_8variants.df'
    dm.manifest_path=OUTPUT/'split_manifest.csv'
    logger=LocalWandbLogger(project='glim',group='A3-v2-V3-50-A800',save_dir=str(OUTPUT),offline=True)
    checkpoint=ModelCheckpoint(dirpath=OUTPUT/'checkpoints',save_top_k=-1,every_n_epochs=1,
        save_last=True,filename='epoch={epoch:02d}-step={step}',auto_insert_metric_name=False)
    trainer=L.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',max_epochs=50,
        default_root_dir=str(OUTPUT),use_distributed_sampler=False,num_sanity_val_steps=0,
        logger=logger,callbacks=[VerifyRestore(saved),Guard(OUTPUT),checkpoint])
    (OUTPUT/'model_config.json').write_text(json.dumps(dict(model.hparams),indent=2))
    (OUTPUT/'run_started.json').write_text(json.dumps(dict(pid=os.getpid(),source=str(SOURCE),
        resume_checkpoint=str(CHECKPOINT),target_total_epochs=50,expected_first_epoch=5,expected_global_step=9940),indent=2))
    trainer.fit(model,datamodule=dm,ckpt_path=str(CHECKPOINT))
    (OUTPUT/'COMPLETE.json').write_text(json.dumps(dict(epochs=50,step=trainer.global_step),indent=2))

if __name__=='__main__': main()
