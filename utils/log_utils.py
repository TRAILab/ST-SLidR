def get_log_prefix(log_groups):
    log_prefix = ''
    if isinstance(log_groups, list):
        for log_group in log_groups:
            log_prefix += f'{log_group}/'
    return log_prefix
