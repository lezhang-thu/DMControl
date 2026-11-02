# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import json
import os
from collections import defaultdict

import torch
from termcolor import colored

FORMAT_CONFIG = {
    'rl': {
        'train': [('episode', 'E', 'int'), ('step', 'S', 'int'),
                  ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
                  ('batch_reward', 'BR', 'float'),
                  ('actor_loss', 'A_LOSS', 'float'),
                  ('critic_loss', 'C_LOSS', 'float'),
                  ('spr_loss', 'SPR_LOSS', 'float')],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise ValueError('invalid format type: %s' % ty)
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    """Write scalar metrics to JSON logs and the console."""
    def __init__(self, log_dir, config='rl'):
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train_info.log'),
                                     formating=FORMAT_CONFIG[config]['train'])
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval_info.log'),
                                    formating=FORMAT_CONFIG[config]['eval'])

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        value = float(value)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')
