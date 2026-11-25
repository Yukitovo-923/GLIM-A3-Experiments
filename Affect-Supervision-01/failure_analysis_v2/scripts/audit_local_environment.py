"""Record local resource feasibility without loading model or EEG data."""

import ctypes
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


OUT=Path(__file__).resolve().parents[1]
WORKSPACE=OUT.parents[1]


def memory_status():
    buffer=ctypes.create_string_buffer(64)
    ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ulong))[0]=64
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"physical_total_bytes":ctypes.cast(ctypes.addressof(buffer)+8,ctypes.POINTER(ctypes.c_ulonglong))[0],
            "physical_available_bytes":ctypes.cast(ctypes.addressof(buffer)+16,ctypes.POINTER(ctypes.c_ulonglong))[0]}


def main():
    gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free,compute_cap",
                        "--format=csv,noheader"],capture_output=True,text=True,check=False)
    disk=shutil.disk_usage(OUT)
    files={
        "raw_eeg_pickle":WORKSPACE/"GLIM_AutoDL_transfer/02_data/zuco_eeg_128ch_1280len.df",
        "flan_t5_weights":WORKSPACE/"GLIM_AutoDL_transfer/04_language_model/flan-t5-large/model.safetensors",
        "v3_checkpoint":WORKSPACE/"topic_recoverability_V100_Pilot/checkpoints/epoch=46-step=93436.ckpt",
        "text_embeddings":WORKSPACE/"A3-v2-results/Text-space-Audit/run-01/text_embeddings.npz",
    }
    result={"memory":memory_status(),"disk_free_bytes":disk.free,
            "gpu_query":gpu.stdout.strip(),"gpu_query_return_code":gpu.returncode,
            "python_packages":{m:importlib.util.find_spec(m) is not None
                               for m in ("numpy","matplotlib","pandas","torch","transformers","lightning")},
            "source_files":{name:{"path":str(path),"exists":path.is_file(),
                                  "size_bytes":path.stat().st_size if path.is_file() else None}
                            for name,path in files.items()}}
    (OUT/"environment.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
