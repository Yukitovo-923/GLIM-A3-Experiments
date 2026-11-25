"""Preserve the official W&B API while also retaining readable local results."""
import json
from pathlib import Path

import torch
from lightning.pytorch.loggers import WandbLogger


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class LocalWandbLogger(WandbLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(kwargs['save_dir'])
        self.generated_texts = []

    def log_table(self, key, columns=None, data=None, dataframe=None, step=None):
        if dataframe is not None:
            rows = [plain(row) for row in dataframe.to_dict(orient='records')]
            name = key.replace('/', '_')
            (self.output_dir / f'{name}.json').write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            dataframe.map(plain).to_csv(self.output_dir / f'{name}.csv', index=False)
            self.generated_texts = [r['Generated Text'] for r in rows if 'Generated Text' in r]
            for row in rows[:3]:
                print('REFERENCE:', row.get('Input Text', ''))
                print('GENERATED:', row.get('Generated Text', ''))
        return super().log_table(key=key, columns=columns, data=data,
                                 dataframe=dataframe, step=step)
