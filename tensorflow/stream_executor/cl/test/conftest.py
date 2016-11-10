import subprocess


def pytest_report_header(config, startdir):
    clinfo_out = subprocess.check_output([
        'clinfo'
    ]).decode('utf-8')
    context = None
    gpu_index = 0
    gpus = ''
    for line in clinfo_out.split('\n'):
        line = line.strip()
        if line.startswith('clCreateContextFromType'):
            context = line.split('NULL, ')[1].split(')')[0]
        if line.startswith('Device Name') and context == 'CL_DEVICE_TYPE_GPU':
            device = line.split('Device Name')[1].strip()
            gpus += 'gpu %s: %s\n' % (gpu_index, device)
            gpu_index += 1
    # print('pytest_report_header')
    return gpus.strip()
