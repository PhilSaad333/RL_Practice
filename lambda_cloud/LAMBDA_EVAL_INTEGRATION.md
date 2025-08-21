# Lambda Cloud Evaluation Integration

🚀 **Submit evaluations from VS Code → run on Lambda → auto-sync results**

## 🎯 Key Features

- **Submit from VS Code**: No more manual SSH commands
- **Auto-batch sizing**: Optimal GPU utilization on Lambda
- **Progress monitoring**: Check job status from local machine  
- **Auto-sync results**: Download results when evaluation completes
- **Job management**: List, monitor, and cleanup evaluation jobs

## 📋 Quick Start

### 1. **Set Lambda IP** (one-time setup)
```bash
python lambda/remote_eval.py set-ip 192.222.52.195
```

### 2. **Test Connection**
```bash  
python lambda/remote_eval.py test
```

### 3. **Submit Evaluation** (Interactive)
```bash
python lambda/vscode_eval.py
```

### 4. **Submit Evaluation** (Command Line)
```bash
python lambda/remote_eval.py submit \
  --training-run run_2025-08-20_03-31-43 \
  --profile full_evaluation
```

### 5. **Monitor Job**
```bash
python lambda/remote_eval.py monitor --job-id eval_20250820_143022_a1b2c3d4
```

### 6. **Sync Results**
```bash
python lambda/remote_eval.py sync --job-id eval_20250820_143022_a1b2c3d4
```

## 🔧 Available Commands

### **Interactive Mode** (Recommended for new users)
```bash
python lambda/vscode_eval.py           # Simple submission
python lambda/vscode_eval.py menu      # Full menu interface
```

### **Command Line Interface**
```bash
# Setup
python lambda/remote_eval.py set-ip <IP>
python lambda/remote_eval.py test

# Job management  
python lambda/remote_eval.py submit --training-run <run> --profile <profile>
python lambda/remote_eval.py list
python lambda/remote_eval.py monitor --job-id <job_id>
python lambda/remote_eval.py sync --job-id <job_id>
python lambda/remote_eval.py cleanup --job-id <job_id>
```

## 📊 Evaluation Profiles

| Profile | Description | Dataset | GPU Usage |
|---------|-------------|---------|-----------|
| `quick_test` | Fast testing | 2% of data | Conservative |
| `full_evaluation` | Complete eval | 100% of data | Auto-optimized |
| `memory_optimized` | Safe full eval | 100% of data | Conservative |
| `high_throughput` | Max performance | 100% of data | Aggressive |
| `debug` | Minimal testing | 0.1% of data | Minimal |

## 🚀 Example Workflows

### **Quick Test on New Training Run**
```bash
python lambda/remote_eval.py submit \
  --training-run run_2025-08-20_15-30-45 \
  --profile quick_test \
  --steps "10,20,final"
```

### **Full Evaluation with Auto-Batch Sizing**
```bash
python lambda/remote_eval.py submit \
  --training-run run_2025-08-20_15-30-45 \
  --profile full_evaluation \
  --batch-size aggressive
```

### **Custom Evaluation**
```bash
python lambda/remote_eval.py submit \
  --training-run run_2025-08-20_15-30-45 \
  --profile memory_optimized \
  --subset-frac 0.1 \
  --batch-size 16
```

## 📁 File Organization

### **Local Job Tracking**
```
~/.lambda_eval_jobs/
├── eval_20250820_143022_a1b2c3d4.json    # Job config & status
├── eval_20250820_150815_b2c3d4e5.json
└── ...
```

### **Synced Results**
```
data/remote_eval_results/
├── eval_20250820_143022_a1b2c3d4/
│   ├── eval_runs/                         # Full evaluation results
│   │   └── run_2025-08-20_03-31-43_gsm8k_r1_template/
│   │       ├── qwen2_5_15_gsm8k_finetuned/
│   │       └── ...
│   └── job_eval_20250820_143022_a1b2c3d4.log  # Evaluation log
└── ...
```

## 🔍 Monitoring & Status

### **Job States**
- **`submitted`**: Job configuration created
- **`running`**: Evaluation in progress on Lambda
- **`finished`**: Evaluation completed
- **`failed`**: Job encountered errors

### **Real-time Monitoring** 
The monitor command shows:
- Job status (running/finished)
- Recent log output (last 10 lines)
- Estimated progress (if available)

### **Job Persistence**
- Job configs stored locally in `~/.lambda_eval_jobs/`
- Survives VS Code restarts and network disconnections
- Lambda IP automatically saved to `~/.lambda_ip`

## ⚙️ Configuration

### **Optional Config File**
Copy `lambda/lambda_eval_config.yaml` to `~/.lambda_eval_config.yaml` to customize:
- Remote paths
- Sync options  
- Timeout settings
- Instance profiles

### **SSH Key Location**
Default: `~/.ssh/lambda_new`
Override: `--ssh-key /path/to/key`

## 🚨 Error Handling

### **Connection Issues**
```bash
# Test connection
python lambda/remote_eval.py test

# Set new IP if instance changed
python lambda/remote_eval.py set-ip <new_ip>
```

### **Job Failures**
```bash
# Check job status and recent logs
python lambda/remote_eval.py monitor --job-id <job_id>

# Cleanup failed job
python lambda/remote_eval.py cleanup --job-id <job_id>
```

### **Sync Issues**
- Check Lambda instance is still running
- Ensure results directory exists on Lambda
- Check SSH key permissions

## 💡 Pro Tips

### **Workflow Integration**
1. **Train on Lambda**: Use existing training workflow
2. **Submit eval from VS Code**: `python lambda/vscode_eval.py`
3. **Continue working**: Job runs in background
4. **Check periodically**: `python lambda/remote_eval.py list`
5. **Sync when done**: Auto-prompted in interactive mode

### **Batch Multiple Evaluations**
```bash
# Submit multiple jobs for different training runs
for run in run_A run_B run_C; do
  python lambda/remote_eval.py submit --training-run $run --profile quick_test
done

# Monitor all jobs
python lambda/remote_eval.py list
```

### **Resource Management**
- Use `quick_test` profile for initial validation
- Use `full_evaluation` for final results
- Use `high_throughput` profile when Lambda instance is dedicated

## 🔮 Future Enhancements

- **Multi-instance support**: Submit to multiple Lambda instances
- **Auto-scaling**: Automatically start/stop Lambda instances
- **Result visualization**: Built-in plotting and analysis
- **Slack/email notifications**: Get notified when jobs complete
- **Cost tracking**: Monitor GPU hours and costs per evaluation

---

**Ready to revolutionize your evaluation workflow!** 🚀

Start with: `python lambda/vscode_eval.py`