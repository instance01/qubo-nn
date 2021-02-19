import copy
import json


class Config:
    def __init__(self):
        with open('simulations.json', 'r') as f:
            self.cfg = json.load(f)

    def _update_cfg(self, base_cfg, new_cfg):
        # We support one level for now.
        for k in new_cfg.keys():
            if k == 'base_cfg' or k == 'desc' or k == 'dataset_id':
                continue
            base_cfg[k].update(new_cfg[k])
        base_cfg['dataset_id'] = new_cfg['dataset_id']

    def get_cfg(self, cfg_id):
        if cfg_id not in self.cfg:
            raise Exception(
                'Error: Key %s does not exist in simulations.json.' % cfg_id
            )

        initial_base_cfg = self.cfg["1"]
        base_cfg = self.cfg[self.cfg[cfg_id].get('base_cfg', cfg_id)]
        # All base configs are based on config "1".
        # This enables backwards compatibility when new options are added.
        self._update_cfg(initial_base_cfg, base_cfg)

        cfg = copy.deepcopy(initial_base_cfg)
        self._update_cfg(cfg, self.cfg[cfg_id])

        cfg['cfg_id'] = cfg_id
        return cfg


if __name__ == '__main__':
    cfg = Config()
    print('\n'.join(cfg.cfg.keys()))
