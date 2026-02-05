import time
import psutil as ps
from datetime import datetime
import csv

try:
  from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetClockInfo,
    NVML_TEMPERATURE_GPU,
    NVML_CLOCK_SM,
  )
  NVML_AVAILABLE = True
except Exception:
  NVML_AVAILABLE = False


# consts
SCENARIO = "idle"
OUTFILE = f"telemetry_{SCENARIO}.csv"
K = 5

Hz = 1.0
period = 1.0 / Hz

# system IO deltas
prev_net = ps.net_io_counters()
prev_disk = ps.disk_io_counters()
prev_t = time.time()

ps.cpu_percent(None)  # warm-up cpu%

# GPU init
gpu_handle = None
if NVML_AVAILABLE:
  try:
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
  except Exception:
    gpu_handle = None


def read_gpu_metrics(handle):
  if handle is None:
    return 0.0, 0.0, 0.0, 0.0, 0.0
  try:
    util = float(nvmlDeviceGetUtilizationRates(handle).gpu)
    mem_used_mb = float(nvmlDeviceGetMemoryInfo(handle).used) / (1024 * 1024)
    power_w = float(nvmlDeviceGetPowerUsage(handle)) / 1000.0
    temp_c = float(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU))
    clock_mhz = float(nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM))
    return util, mem_used_mb, power_w, temp_c, clock_mhz
  except Exception:
    return 0.0, 0.0, 0.0, 0.0, 0.0


# -------------------------
# header
# -------------------------
f = open(OUTFILE, "w", newline="", encoding="utf-8")
writer = csv.writer(f)

header = [
  "ts","scenario","cpu","ram",
  "net_in_bps","net_out_bps",
  "disk_read_bps","disk_write_bps",
  "gpu_util","gpu_mem_used_mb","gpu_power_w","gpu_temp_c","gpu_clock_mhz"
]

# per-process: 4 features only (cheapest)
for j in range(1, K + 1):
  header += [f"p{j}_cpu", f"p{j}_rss_mb", f"p{j}_threads", f"p{j}_age_s"]

writer.writerow(header)

# -------------------------
# state
# -------------------------
i = 0
proc_warm = False

try:
  while True:
    t0 = time.time()

    # system metrics
    cpu = ps.cpu_percent(None)
    ram = ps.virtual_memory().percent

    # system net/disk throughput
    now_net = ps.net_io_counters()
    now_disk = ps.disk_io_counters()
    now_t = time.time()

    dt_io = now_t - prev_t
    if dt_io <= 0:
      net_in_bps = net_out_bps = 0.0
      disk_read_bps = disk_write_bps = 0.0
    else:
      net_out_bps = (now_net.bytes_sent - prev_net.bytes_sent) / dt_io
      net_in_bps = (now_net.bytes_recv - prev_net.bytes_recv) / dt_io
      disk_read_bps = (now_disk.read_bytes - prev_disk.read_bytes) / dt_io
      disk_write_bps = (now_disk.write_bytes - prev_disk.write_bytes) / dt_io

    prev_net = now_net
    prev_disk = now_disk
    prev_t = now_t

    # warm-up per-process cpu_percent
    if not proc_warm:
      for p in ps.process_iter():
        try:
          p.cpu_percent(None)
        except (ps.AccessDenied, ps.NoSuchProcess):
          pass
      proc_warm = True
      continue

    # collect per-process candidates (cheap)
    items = []
    for p in ps.process_iter(attrs=["pid", "name"]):
      try:
        pid = p.info["pid"]
        name = p.info["name"]

        if pid == 0 or name == "System Idle Process":
          continue

        cpu_p = p.cpu_percent(None)
        rss_mb = p.memory_info().rss / (1024 * 1024)
        threads = p.num_threads()
        age_s = time.time() - p.create_time()

        items.append((cpu_p, rss_mb, threads, age_s))

      except (ps.AccessDenied, ps.NoSuchProcess):
        continue

    top = sorted(items, key=lambda x: x[0], reverse=True)[:K]

    proc_features = []
    for (cpu_p, rss_mb, threads, age_s) in top:
      proc_features.extend([cpu_p, rss_mb, threads, age_s])

    while len(proc_features) < K * 4:
      proc_features.extend([0.0] * 4)

    # GPU
    gpu_util, gpu_mem_mb, gpu_power_w, gpu_temp_c, gpu_clock_mhz = read_gpu_metrics(gpu_handle)

    ts = datetime.now().isoformat(timespec="seconds")
    row = [
      ts, SCENARIO, cpu, ram,
      net_in_bps, net_out_bps,
      disk_read_bps, disk_write_bps,
      gpu_util, gpu_mem_mb, gpu_power_w, gpu_temp_c, gpu_clock_mhz
    ]
    row += proc_features
    writer.writerow(row)

    dt = time.time() - t0
    time.sleep(max(0.0, period - dt))

    i += 1
    if i == 49:
      f.flush()
      i = 0

finally:
  try:
    if gpu_handle is not None and NVML_AVAILABLE:
      nvmlShutdown()
  except Exception:
    pass
  f.close()
