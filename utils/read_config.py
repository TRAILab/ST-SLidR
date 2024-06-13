from pathlib import Path

from omegaconf import OmegaConf


def get_config_list(file):
    config_list = []
    with open(file, "r") as f:
        config = OmegaConf.load(f)
    # Base config loading
    if '_base_config' in config:
        config_list.extend(get_config_list(config["_base_config"]))
    config_list.append(config)
    return config_list


def generate_config(file, mod_file=None, extra_tag=None, random_seed=None):
    config_list = get_config_list(file)

    if mod_file is not None:
        with open(mod_file, "r") as f:
            mod_config = OmegaConf.load(f)
        config_list.append(mod_config)

    config = OmegaConf.merge(*config_list)

    if extra_tag:
        config["extra_tag"] = extra_tag

    if random_seed:
        config["random_seed"] = random_seed

    # Add experiment name to config
    file_parts = Path(file).parts
    print(file_parts)
    if "exps" in file_parts:
        exp_group_id = file_parts.index("exps") + 1
        config["experiment"] = file_parts[exp_group_id]


    return OmegaConf.to_container(config)


def print_config(cfg, pre='cfg'):
    for key, val in cfg.items():
        if isinstance(cfg[key], dict):
            print('\n%s.%s = odict()' % (pre, key))
            print_config(cfg[key], pre=pre + '.' + key)
            continue
        print('%s.%s: %s' % (pre, key, val))
