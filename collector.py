import time
import psutil as ps
from datetime import datetime
import csv


#consts
SCENARIO = "idle"
OUTFILE = f"telemetry_{SCENARIO}.csv"
K = 5 #number of top procceses


Hz = 1.0 #how many times per second we wanna iterate the cycle
period = 1.0 / Hz #how many seconds lasts one cycle iter

#initializing vars
prev_net = ps.net_io_counters()
prev_disk = ps.disk_io_counters()
prev_t = time.time()
#---------------------

ps.cpu_percent(None) #warming-up

#---------------------
#header
f = open(OUTFILE, "w", newline="", encoding="utf-8")
writer = csv.writer(f)
header = ["ts","scenario","cpu","ram","net_in_bps","net_out_bps","disk_read_bps","disk_write_bps"]

for j in range(1, K+1):
  header += [f"p{j}_cpu", f"p{j}_rss", f"p{j}_threads", f"p{j}_age"]
writer.writerow(header)

#---------------------
i = 0 #counter for .flush()
proc_warm =False

try:
  
  while True:
    t0 = time.time() #moment of starting of cycle

    cpu = ps.cpu_percent(None)
    ram = ps.virtual_memory().percent
    

    now_net = ps.net_io_counters()
    now_disk = ps.disk_io_counters()
    now_t = time.time()

    dt_io = now_t - prev_t
  
    if dt_io <= 0: 
      net_in_bps = 0.0
      net_out_bps = 0.0
      disk_read_bps = 0.0
      disk_write_bps = 0.0
    else:
      net_out_bps = (now_net.bytes_sent - prev_net.bytes_sent) / dt_io
      net_in_bps = (now_net.bytes_recv - prev_net.bytes_recv) / dt_io
  
      disk_read_bps = (now_disk.read_bytes - prev_disk.read_bytes) / dt_io
      disk_write_bps = (now_disk.write_bytes - prev_disk.write_bytes) / dt_io


    prev_net = now_net
    prev_disk = now_disk
    prev_t = now_t

    if not proc_warm:
      for p in ps.process_iter():
        try:
          p.cpu_percent(None)
        except (ps.AccessDenied, ps.NoSuchProcess):
          pass
      proc_warm = True
      continue
    #------------------------
    # preparing to take top K-proc  
    items = []

    for p in ps.process_iter(attrs=["pid", "name"]):
      try:
        if p.info["name"] == "System Idle Process":
          continue
        cpu_p = p.cpu_percent(None)
        rss = p.memory_info().rss / (1024*1024)
        threads = p.num_threads()
        age = time.time() - p.create_time()
        items.append((cpu_p, rss, threads, age))
      except (ps.AccessDenied, ps.NoSuchProcess):
        continue
    
    #-----------------------------
    #sorting our top K-procses

    top = sorted(items, reverse=True)[:K]
   
    #vectorizing top K-proc data

    proc_features = []

    for (cpu_p, rss, threads, age) in top:
      proc_features.extend([cpu_p, rss, threads, age])
    
    while len(proc_features) < K * 4: 
      proc_features.extend([0.0, 0.0, 0.0, 0.0]) #makes sure that it always not less than K*4, needed for correct work of DL model
    
    #--------------------------------


    ts = datetime.now().isoformat(timespec="seconds")
    row = [ts, SCENARIO, cpu, ram, net_in_bps, net_out_bps, disk_read_bps, disk_write_bps]
    row += proc_features
    writer.writerow(row)


    dt = time.time() - t0 #how much time took operation in cycle
    time.sleep(max(0.0, period - dt))
    i+=1
    if i == 49:
      f.flush() #forcfully writes in file from buffer
      i = 0


finally:
  f.close()
    

