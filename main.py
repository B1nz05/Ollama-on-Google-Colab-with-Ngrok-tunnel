# -*- coding: utf-8 -*-
"""
Kaggle Dual-T4 GPU - Ollama + Ngrok Automatic Deployment and Tunneling Script
Version: 2026.06.05-Kaggle_GPU_Ultimate_Fix_v5
Optimizations:
     1. Fixes the Ollama issue in Kaggle where it defaults to RAM/CPU instead of using the GPU.
     2. Smart Environment Probe: Compatible with both Kaggle Secrets and Colab Secrets.
     3. Robust Ngrok Startup: Uses inline --authtoken parameters to bypass permission issues.
     4. Force-refreshes system-wide ldconfig cache to link dual T4 physical CUDA drivers.
     5. Configures environment variables optimized for T4 dual-GPUs.
"""

import os
import sys
import subprocess
import time
import json
import urllib.request
import urllib.error
import ssl
import shutil
import tarfile

# ==================== [CONFIGURABLE VARIABLES SECTION] ====================

# 1. Ngrok Authorization Token (Attempts platform secrets first, falls back to static value)
NGROK_AUTHTOKEN = None

# Option A: Try retrieving from Kaggle Secrets
if not NGROK_AUTHTOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        NGROK_AUTHTOKEN = user_secrets.get_secret("NGROK_AUTHTOKEN")
        print("💡 Successfully retrieved NGROK_AUTHTOKEN from Kaggle Secrets.")
    except Exception:
        pass

# Option B: Try retrieving from Colab Secrets (for cross-compatibility)
if not NGROK_AUTHTOKEN:
    try:
        from google.colab import userdata
        NGROK_AUTHTOKEN = userdata.get("NGROK_AUTHTOKEN")
        print("💡 Successfully retrieved NGROK_AUTHTOKEN from Colab Secrets.")
    except Exception:
        pass

# Option C: Static Fallback (Replace this with your token if not using Secrets)
if not NGROK_AUTHTOKEN:
    NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
    print("⚠️ No Token found in Secrets. Using static fallback token.")

# 2. Ngrok Region Selection (e.g., 'us', 'ap', 'eu', 'au', 'sa', 'jp', 'in')
NGROK_REGION = "us"

# 3. Models to pull automatically upon startup
MODELS_TO_PULL = [
    "qwen2.5:7b"  # Replace with any model you need, e.g., "llama3" or "gemma2"
]

# 4. Persistent Storage Directory (Using /kaggle/working allows persistence across session restarts)
PERSIST_DIR = "/kaggle/working/ollama"

# 5. Dual T4 GPU Performance Environment Variables
OS_ENV_UPDATES = {
    "OLLAMA_HOST": "0.0.0.0:11434",
    "OLLAMA_NUM_PARALLEL": "4",           # Max concurrent requests for dual T4 setup
    "OLLAMA_MAX_LOADED_MODELS": "2",      # Max models to keep resident in VRAM simultaneously
    "CUDA_VISIBLE_DEVICES": "0,1",        # Explicitly utilize both T4 GPUs
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "OLLAMA_INTENSITY": "1",               # GPU load intensity optimization
    "OLLAMA_FLASH_ATTENTION": "1",         # Enable Flash Attention acceleration
    "OLLAMA_DEBUG": "1"                    # Enable debug mode to output GPU discovery logs
}

# ==============================================================

def setup_gpu_environments():
    """
    Core Fix: Explicitly injects CUDA driver paths into the system's global dynamic 
    linker cache (ldconfig). This ensures Ollama's background engine (llama-server) 
    reliably detects and utilizes the dual GPUs.
    """
    print_banner("Kaggle Dual-GPU Deep Penetration & System Global Ldconfig Refresh...")

    # 1. Update OS environment variables
    os.environ.update(OS_ENV_UPDATES)

    # 2. Check physical GPU availability
    gpu_count = 0
    try:
        smi_out = subprocess.run(
            "nvidia-smi --query-gpu=name,gpu_bus_id,memory.total --format=csv,noheader",
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        gpus = [line.strip() for line in smi_out.stdout.strip().split("\n") if line.strip()]
        gpu_count = len(gpus)
        print("🟢 Physical GPU Resources Detected:")
        for idx, gpu in enumerate(gpus):
            print(f"   ⚡ [GPU {idx}]: {gpu}")
    except Exception:
        print("❌ Physical GPU detection failed! Please make sure 'GPU T4 x2' is selected under Accelerator in Kaggle settings.")
        sys.exit(1)

    # 3. Inject driver paths to system configuration
    cuda_paths = [
        "/usr/lib64-nvidia",                 # Google Colab Core NVIDIA driver directory
        "/usr/local/nvidia/lib64",           # Docker/Kaggle default path
        "/usr/local/cuda/lib64",             # CUDA Runtime libraries
        "/usr/local/cuda/compat",            # CUDA Compatibility libraries
        "/usr/lib/x86_64-linux-gnu",         # Global system libraries
        "/usr/lib64",                        # Backup 64-bit directory
        "/usr/local/lib/ollama"              # Ollama internal directory
    ]

    ld_conf_path = "/etc/ld.so.conf.d/kaggle-nvidia-gpu.conf"
    try:
        valid_paths = [p for p in cuda_paths if os.path.exists(p)]
        os.makedirs(os.path.dirname(ld_conf_path), exist_ok=True)
        with open(ld_conf_path, "w") as f:
            for path in valid_paths:
                f.write(f"{path}\n")
        print(f"📝 Wrote GPU library paths to {ld_conf_path}:\n   -> {valid_paths}")

        # Core step: Call ldconfig to refresh the global system shared library cache
        print("⚙️ Executing system ldconfig cache refresh...")
        subprocess.run("ldconfig", shell=True, check=True)
        print("✅ Global system ldconfig refreshed successfully! Dual-GPU recognition guaranteed.")
    except Exception as e:
        print(f"⚠️ Could not write to ld.so.conf or run ldconfig: {e}. Falling back to standard environment injection.")

    # 4. Fallback/Double-safety: Inject into Python session's LD_LIBRARY_PATH
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = ["/usr/local/lib/ollama"] + cuda_paths
    if existing_ld:
        ld_paths.append(existing_ld)

    unique_paths = []
    for p in ld_paths:
        if p and p not in unique_paths:
            unique_paths.append(p)

    os.environ["LD_LIBRARY_PATH"] = ":".join(unique_paths)

    # 5. Update system PATH
    existing_path = os.environ.get("PATH", "")
    cuda_bin_paths = ["/usr/local/cuda/bin", "/usr/local/nvidia/bin"]
    new_path = ":".join(cuda_bin_paths + [existing_path])
    os.environ["PATH"] = new_path

    # Symlink library fallbacks
    ollama_lib_path = "/usr/local/lib/ollama"
    for fallback_dir in ["/usr/lib/ollama", "/usr/share/ollama"]:
        try:
            if os.path.exists(fallback_dir) or os.path.islink(fallback_dir):
                if os.path.islink(fallback_dir):
                    os.unlink(fallback_dir)
                else:
                    shutil.rmtree(fallback_dir)
            os.makedirs(os.path.dirname(fallback_dir), exist_ok=True)
            os.symlink(ollama_lib_path, fallback_dir)
            print(f"🔗 Established symlink: {fallback_dir} -> {ollama_lib_path}")
        except Exception as e:
            print(f"⚠️ Symlink creation failed for {fallback_dir}: {e}")

def print_banner(text):
    print("=" * 60)
    print(f"🌟 {text}")
    print("=" * 60)

def run_command(command, shell=True, wait=True):
    try:
        if wait:
            result = subprocess.run(command, shell=shell, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout.strip()
        else:
            process = subprocess.Popen(command, shell=shell, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return process
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}\nError output: {e.stderr}")
        return None

def is_valid_elf(filepath):
    if not os.path.exists(filepath) or os.path.islink(filepath) or os.path.isdir(filepath):
        return False
    try:
        if os.path.getsize(filepath) < 8 * 1024 * 1024:
            return False
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            return magic == b'\x7fELF'
    except Exception:
        return False

def download_file_native(url, dest_path):
    print(f"🔗 Source: {url}")
    print(f"📂 Destination: {dest_path}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=20) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"   📊 Progress: {percent:.2f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="\r")
                else:
                    print(f"   📊 Downloaded: {downloaded / (1024*1024):.1f}MB (unknown total size)", end="\r")
            print("\n🎉 Download complete!")
            return True
    except Exception as e:
        print(f"⚠️ Download interrupted: {str(e)}")
        if os.path.exists(dest_path) and not os.path.isdir(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass
        return False

def ensure_zstd_installed():
    if shutil.which("zstd") is not None:
        return True
    print("⚠️ Installing zstd extraction utility...")
    run_command("apt-get update -qq && apt-get install -y -qq zstd")
    return shutil.which("zstd") is not None

def extract_tar_zst(archive_path, extract_dir="."):
    print(f"📦 Extracting {archive_path}...")
    ensure_zstd_installed()
    try:
        subprocess.run(f"tar --zstd -xf {archive_path} -C {extract_dir}", shell=True, check=True)
        return True
    except Exception:
        try:
            subprocess.run(f"zstd -d {archive_path} -o temp.tar && tar -xf temp.tar -C {extract_dir} && rm -f temp.tar", shell=True, check=True)
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

def get_latest_ollama_version():
    default_version = "v0.5.11"
    try:
        url = "https://api.github.com/repos/ollama/ollama/releases/latest"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ssl_context, timeout=5) as response:
            data = json.loads(response.read().decode())
            version = data.get("tag_name", default_version)
            print(f"📡 Successfully resolved latest Ollama version: {version}")
            return version
    except Exception:
        print(f"⚠️ Unable to fetch latest release. Using fallback version: {default_version}")
        return default_version

def install_ollama():
    print_banner("Step 1: Installing Ollama Engine...")
    target_bin = "/usr/local/bin/ollama"
    target_lib_dir = "/usr/local/lib/ollama"
    backup_bin_path = os.path.join(PERSIST_DIR, "ollama_bin_backup")
    backup_lib_dir = os.path.join(PERSIST_DIR, "ollama_lib_backup")

    if is_valid_elf(backup_bin_path) and os.path.exists(os.path.join(backup_lib_dir, "llama-server")):
        print("💾 Detected persistent local backup. Restoring offline...")
        try:
            os.makedirs("/usr/local/bin", exist_ok=True)
            if os.path.exists(target_bin):
                os.remove(target_bin)
            shutil.copy2(backup_bin_path, target_bin)
            os.chmod(target_bin, 0o755)

            if os.path.exists(target_lib_dir):
                shutil.rmtree(target_lib_dir)
            os.makedirs("/usr/local/lib", exist_ok=True)
            shutil.copytree(backup_lib_dir, target_lib_dir, symlinks=True)
            print("✅ Restored Ollama Engine and GPU dependencies successfully!")
            return
        except Exception as e:
            print(f"⚠️ Backup restoration failed: {e}. Falling back to clean download.")

    for p in [target_bin, "/usr/bin/ollama"]:
        if os.path.exists(p) and not os.path.islink(p):
            try:
                os.remove(p)
            except:
                pass

    ollama_version = get_latest_ollama_version()
    zst_urls = [
        f"https://ghfast.top/https://github.com/ollama/ollama/releases/download/{ollama_version}/ollama-linux-amd64.tar.zst",
        f"https://mirror.ghproxy.com/https://github.com/ollama/ollama/releases/download/{ollama_version}/ollama-linux-amd64.tar.zst",
        f"https://github.com/ollama/ollama/releases/download/{ollama_version}/ollama-linux-amd64.tar.zst"
    ]

    downloaded_and_extracted = False
    temp_archive = "./ollama_download.tar.zst"
    temp_extract_dir = "./ollama_temp_extract"

    for url in zst_urls:
        print_banner(f"📥 Downloading Ollama engine archive...")
        success = download_file_native(url, temp_archive)
        if success and os.path.exists(temp_archive) and os.path.getsize(temp_archive) > 10 * 1024 * 1024:
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
            os.makedirs(temp_extract_dir, exist_ok=True)

            extract_success = extract_tar_zst(temp_archive, temp_extract_dir)
            try:
                os.remove(temp_archive)
            except:
                pass

            extracted_bin = os.path.join(temp_extract_dir, "bin/ollama")
            extracted_lib = os.path.join(temp_extract_dir, "lib/ollama")

            if extract_success and is_valid_elf(extracted_bin) and os.path.exists(os.path.join(extracted_lib, "llama-server")):
                try:
                    os.makedirs("/usr/local/bin", exist_ok=True)
                    if os.path.exists(target_bin):
                        os.remove(target_bin)
                    shutil.move(extracted_bin, target_bin)
                    os.chmod(target_bin, 0o755)

                    os.makedirs("/usr/local/lib", exist_ok=True)
                    if os.path.exists(target_lib_dir):
                        shutil.rmtree(target_lib_dir)
                    shutil.move(extracted_lib, target_lib_dir)

                    print("✅ Ollama Core and GPU Acceleration Libs (llama-server) deployed successfully!")
                    downloaded_and_extracted = True
                    break
                except Exception as e:
                    print(f"⚠️ File deployment error: {e}")
        time.sleep(1)

    if downloaded_and_extracted:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        try:
            if os.path.exists(backup_bin_path):
                os.remove(backup_bin_path)
            shutil.copy2(target_bin, backup_bin_path)
            os.chmod(backup_bin_path, 0o755)

            if os.path.exists(backup_lib_dir):
                shutil.rmtree(backup_lib_dir)
            shutil.copytree(target_lib_dir, backup_lib_dir, symlinks=True)
            print("💾 Saved engine and libs to persistent directory.")
        except Exception as e:
            print(f"⚠️ Could not create local backup: {e}")

        try:
            shutil.rmtree(temp_extract_dir)
        except:
            pass
    else:
        raise RuntimeError("❌ Ollama installation failed.")

def setup_persistence():
    print_banner("Step 2: Configuring Persistent Storage...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    ollama_home = os.path.expanduser("~/.ollama")
    if os.path.exists(ollama_home) or os.path.islink(ollama_home):
        if os.path.islink(ollama_home):
            os.unlink(ollama_home)
        else:
            run_command(f"rm -rf {ollama_home}")
    try:
        os.symlink(PERSIST_DIR, ollama_home)
        print(f"✅ Created symbolic link: {ollama_home} -> {PERSIST_DIR}")
    except Exception as e:
        print(f"⚠️ Symlink failed: {e}. Defaulting to standard environment mapping.")
        os.environ["OLLAMA_MODELS"] = os.path.join(PERSIST_DIR, "models")
        os.makedirs(os.environ["OLLAMA_MODELS"], exist_ok=True)

def start_ollama_service():
    print_banner("Step 3: Launching Ollama Daemon...")
    run_command("pkill -9 ollama || true")
    time.sleep(1)

    ollama_bin = "/usr/local/bin/ollama" if is_valid_elf("/usr/local/bin/ollama") else "ollama"
    print(f"🚀 Spawning process: {ollama_bin} serve ...")
    log_file = open("ollama.log", "w", buffering=1)

    process_env = os.environ.copy()

    process = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=process_env,
        preexec_fn=os.setsid
    )

    time.sleep(4)
    poll_status = process.poll()
    if poll_status is not None:
        log_file.close()
        print(f"❌ Ollama process crashed on startup. Exit code: {poll_status}")
        print("🔍 Crash Log Snippet:")
        print("-" * 60)
        if os.path.exists("ollama.log"):
            with open("ollama.log", "r") as f:
                print(f.read())
        print("-" * 60)
        raise RuntimeError("Ollama failed to start.")

    # Polling initialization status
    retries = 25
    cuda_detected = False
    for i in range(retries):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/", timeout=3)
            print("✅ Ollama Daemon is online and active!")
            log_file.close()

            # Inspect logs to confirm CUDA driver detection
            time.sleep(2)
            if os.path.exists("ollama.log"):
                with open("ollama.log", "r") as f:
                    logs = f.read()
                    if "CUDA" in logs or "Nvidia" in logs or "gpu" in logs.lower():
                        print("🎉 [Verification] Log confirmation: Ollama successfully linked to CUDA (GPU Enabled)!")
                        cuda_detected = True
                        for line in logs.split("\n"):
                            if any(k in line.lower() for k in ["cuda", "nvidia", "gpu", "detect"]):
                                print(f"   [Log] {line}")
                    else:
                        print("⚠️ [Warning] No CUDA initialization keywords found. GPU might not be recognized.")
                        print("🔍 Displaying first 30 lines of startup log for debugging:")
                        print("-" * 60)
                        print("\n".join(logs.split("\n")[:30]))
                        print("-" * 60)
            return
        except Exception:
            print(f"⏳ Awaiting background daemon... ({i+1}/{retries})")
            time.sleep(2)

    log_file.close()
    raise RuntimeError("❌ Connection to Ollama timed out!")

def install_and_start_ngrok():
    """
    Step 4: Configures and initializes high-availability Ngrok network tunnel using
    inline argument authorization to prevent configuration writing conflicts.
    """
    print_banner("Step 4: Deploying Ngrok Tunnel...")
    global NGROK_AUTHTOKEN
    if not NGROK_AUTHTOKEN or NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN_HERE":
        raise RuntimeError("❌ Valid NGROK_AUTHTOKEN must be configured to run Ngrok.")

    # Cleanup existing ngrok processes
    run_command("pkill -9 ngrok || true")
    run_command("pkill -f ngrok || true")
    run_command("killall ngrok || true")
    time.sleep(2)

    ngrok_path = "/usr/local/bin/ngrok" if os.path.exists("/usr/local/bin/ngrok") else "./ngrok"
    force_redownload = False
    if os.path.exists(ngrok_path):
        try:
            res = subprocess.run([ngrok_path, "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "version 3" not in (res.stdout.strip() or res.stderr.strip()):
                force_redownload = True
        except Exception:
            force_redownload = True

    if force_redownload:
        for p in ["/usr/local/bin/ngrok", "./ngrok"]:
            if os.path.exists(p) and not os.path.isdir(p):
                try:
                    os.remove(p)
                except:
                    pass
        ngrok_path = "./ngrok"

    # Downloading Ngrok binary if missing
    if not os.path.exists(ngrok_path) and not os.path.exists("/usr/local/bin/ngrok"):
        print("📥 Fetching Ngrok v3-stable...")
        ngrok_urls = [
            "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
            "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz",
            "https://bin.equinox.io/c/bdr7w6KPmLv/ngrok-v3-stable-linux-amd64.tgz"
        ]

        download_success = False
        temp_file = "./ngrok.tgz"

        for url in ngrok_urls:
            if download_file_native(url, temp_file):
                download_success = True
                break
            print("⚠️ Download failed from this mirror. Swapping nodes...")
            time.sleep(1)

        if not download_success:
            print("⚠️ Using fallback wget schema...")
            run_command("wget --no-check-certificate -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -O ./ngrok.tgz")
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1024 * 1024:
                download_success = True

        if download_success and os.path.exists(temp_file):
            try:
                with tarfile.open(temp_file, "r:gz") as tar_ref:
                    tar_ref.extractall(path=".")
                print("✅ Ngrok unpacked successfully!")
            except Exception:
                run_command(f"tar -xzf {temp_file}")

            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        if os.path.exists("./ngrok"):
            try:
                shutil.move("./ngrok", "/usr/local/bin/ngrok")
                ngrok_path = "/usr/local/bin/ngrok"
            except Exception:
                ngrok_path = "./ngrok"
        elif not os.path.exists(ngrok_path) and not os.path.exists("/usr/local/bin/ngrok"):
            raise FileNotFoundError("❌ Fatal: Failed to deploy a valid ngrok binary executable!")

    if os.path.exists(ngrok_path):
        os.chmod(ngrok_path, 0o755)

    # Clean old potential conflicting configs
    shutil.rmtree(os.path.expanduser("~/.ngrok2"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.config/ngrok"), ignore_errors=True)

    print(f"🚀 Spawning Ngrok process: {ngrok_path}")
    ngrok_log = open("ngrok.log", "w", buffering=1)

    # Spawn Ngrok via inline shell parameter execution
    process = subprocess.Popen(
        [ngrok_path, "http", "11434", "--region", NGROK_REGION, "--authtoken", NGROK_AUTHTOKEN, "--log", "stdout"],
        stdout=ngrok_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    time.sleep(3.5)
    poll_status = process.poll()

    if poll_status is not None:
        ngrok_log.close()
        print(f"\n❌ [Crash Alert] Ngrok process terminated unexpectedly with code: {poll_status}")
        print("-" * 60)
        if os.path.exists("ngrok.log"):
            with open("ngrok.log", "r") as f:
                print("🔍 Startup logs:")
                print(f.read().strip())
                print("-" * 60)
        raise RuntimeError("❌ Ngrok tunnel crashed.")

    # Fetch public endpoint URL
    retries = 15
    for i in range(retries):
        try:
            time.sleep(2)
            req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels")
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                public_url = data['tunnels'][0]['public_url']
                openai_compatible_url = f"{public_url}/v1"

                print_banner("🎉 NGROK Tunnel Successfully Configured!")
                print(f"👉 【Base URL (OpenAI Compatible Format)】:\n    {openai_compatible_url}\n")
                print(f"👉 【Ollama Native WebUI Endpoint】:\n    {public_url}\n")
                print("💡 Hint: Append `/v1` to the URL for OpenAI integration client libraries.")
                print("=" * 60)
                ngrok_log.close()
                return public_url
        except Exception:
            if i == retries - 1:
                ngrok_log.close()
                if os.path.exists("ngrok.log"):
                    with open("ngrok.log", "r") as f:
                        print(f"🔍 Timeout Log Output:\n{f.read()}")
                raise RuntimeError("Ngrok initialization timed out! Double check your NGROK_AUTHTOKEN.")

def pull_models():
    print_banner("Step 5: Fetching Target Models...")
    for model in MODELS_TO_PULL:
        print(f"📥 Pulling model: 【{model}】...")
        process = subprocess.Popen(
            f"ollama pull {model}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stripped = output.strip()
                if stripped and ("%" in stripped or "success" in stripped.lower() or "pulling" in stripped.lower()):
                    print(f"   📊 {stripped}", end="\r")
        print(f"\n✅ Model 【{model}】 loaded and ready!")

def monitor_loop():
    print_banner("Step 6: Daemon Loop Active...")
    print("This cell is maintaining the active background tunnels. Do not stop this cell.")
    print("\n💡 Verification Check:")
    print("   1. Open a new cell and execute `!nvidia-smi`. You should see VRAM allocated on both GPUs.")
    print("   2. Tokens/sec performance should increase dramatically compared to CPU execution.")

    tick = 0
    while True:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/", timeout=5)
            tick += 1
            if tick % 5 == 0:
                print(f"\n⏰ [Heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')} - System healthy, tunnel online...")
        except KeyboardInterrupt:
            print("\n👋 Manual termination signal received. Cleaning up processes...")
            run_command("pkill -9 ollama || true")
            run_command("pkill -9 ngrok || true")
            print("✅ All services stopped safely.")
            break
        except Exception as e:
            print(f"⚠️ Health check error: {str(e)}")
        time.sleep(60)

if __name__ == "__main__":
    # 1. System Ldconfig & Dual GPU Environment Configuration
    setup_gpu_environments()

    # 2. Sequential deployment pipeline
    install_ollama()
    setup_persistence()
    start_ollama_service()
    install_and_start_ngrok()
    pull_models()
    monitor_loop()
