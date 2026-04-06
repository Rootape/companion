# Guia Colab — Treino de Wake Word "Anfitrião"

Execute as células **nesta ordem exata**. Todas as células novas devem ser criadas manualmente no notebook.

---

## CÉLULA 1 — Instalação (original do notebook, sem alterações)

```python
# install piper-sample-generator (currently only supports linux systems)
!git clone https://github.com/rhasspy/piper-sample-generator
!wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
!pip install piper-phonemize
!pip install webrtcvad

# install openwakeword (full installation to support training)
!git clone https://github.com/dscripka/openwakeword
!pip install -e ./openwakeword
!cd openwakeword

# install other dependencies
!pip install mutagen==1.47.0
!pip install torchinfo==1.8.0
!pip install torchmetrics==1.2.0
!pip install speechbrain==0.5.14
!pip install audiomentations==0.33.0
!pip install torch-audiomentations==0.11.0
!pip install acoustics==0.2.6
!pip install tensorflow-cpu==2.8.1
!pip install tensorflow_probability==0.16.0
!pip install onnx_tf==1.10.0
!pip install pronouncing==0.2.0
!pip install datasets==2.14.6
!pip install deep-phonemizer==0.0.19

# Download required models (workaround for Colab)
import os
os.makedirs("./openwakeword/openwakeword/resources/models")
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
!wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite
```

---

## CÉLULA 2 — Patches e amostras (NOVA — rodar logo após Célula 1)

> Faça upload do `wake_word_samples.zip` no Colab antes de rodar esta célula.

```python
import os, sys, shutil, zipfile

# ── Fix 1: torchaudio.set_audio_backend removido nas versões novas ──
filepath = "/usr/local/lib/python3.12/dist-packages/torch_audiomentations/utils/io.py"
with open(filepath, 'r') as f: content = f.read()
content = content.replace('torchaudio.set_audio_backend("soundfile")', 'pass')
with open(filepath, 'w') as f: f.write(content)

# ── Fix 2: torchaudio.info removido nas versões novas ──
with open(filepath, 'r') as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    if 'info = torchaudio.info(file_path)' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f'{indent}import soundfile as _sf\n')
        new_lines.append(f'{indent}_sf_info = _sf.info(str(file_path))\n')
        new_lines.append(f'{indent}class _InfoProxy:\n')
        new_lines.append(f'{indent}    num_frames = _sf_info.frames\n')
        new_lines.append(f'{indent}    sample_rate = _sf_info.samplerate\n')
        new_lines.append(f'{indent}info = _InfoProxy()\n')
    else:
        new_lines.append(line)
with open(filepath, 'w') as f:
    f.writelines(new_lines)

# ── Fix 3: torch.load weights_only no deep-phonemizer ──
filepath2 = "/usr/local/lib/python3.12/dist-packages/dp/model/model.py"
with open(filepath2, 'r') as f: content2 = f.read()
content2 = content2.replace(
    'torch.load(checkpoint_path, map_location=device)',
    'torch.load(checkpoint_path, map_location=device, weights_only=False)'
)
with open(filepath2, 'w') as f: f.write(content2)

# ── Fix 4: generate_samples não existe mais no piper-sample-generator ──
with open('openwakeword/openwakeword/train.py', 'r') as f: content3 = f.read()
if "sys.path.insert(0, '/content/piper-sample-generator')\n" not in content3:
    content3 = "import sys\nsys.path.insert(0, '/content/piper-sample-generator')\n" + content3
    with open('openwakeword/openwakeword/train.py', 'w') as f: f.write(content3)

with open('/content/piper-sample-generator/generate_samples.py', 'w') as f:
    f.write("def generate_samples(*args, **kwargs): pass\n")

# ── Extrair amostras e popular diretórios ──
with zipfile.ZipFile("wake_word_samples.zip", "r") as z:
    z.extractall("wake_word_samples_extracted")

samples = sorted(os.listdir("wake_word_samples_extracted/positive"))
positive_train = "./my_custom_model/anfitriao/positive_train"
positive_test  = "./my_custom_model/anfitriao/positive_test"
os.makedirs(positive_train, exist_ok=True)
os.makedirs(positive_test, exist_ok=True)

n_train = int(len(samples) * 0.8)
for f in samples[:n_train]:
    shutil.copy(f"wake_word_samples_extracted/positive/{f}", f"{positive_train}/{f}")
for f in samples[n_train:]:
    shutil.copy(f"wake_word_samples_extracted/positive/{f}", f"{positive_test}/{f}")

print(f"Patches aplicados.")
print(f"Amostras — Train: {len(os.listdir(positive_train))} | Test: {len(os.listdir(positive_test))}")
```

---

## CÉLULA 3 — Imports (original do notebook, sem alterações)

```python
import os
import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm
```

---

## CÉLULA 4 — Download MIT RIRs (original do notebook, sem alterações)

```python
output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

for row in tqdm(rir_dataset):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
```

---

## CÉLULA 5 — Download background audio (original do notebook, sem alterações)

```python
if not os.path.exists("audioset"):
    os.mkdir("audioset")

fname = "bal_train09.tar"
out_dir = f"audioset/{fname}"
link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
!wget -O {out_dir} {link}
!cd audioset && tar -xvf bal_train09.tar

output_dir = "./audioset_16k"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})
audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
for row in tqdm(audioset_dataset):
    name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

n_hours = 1
for i in tqdm(range(n_hours*3600//30)):
    row = next(fma_dataset)
    name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    i += 1
    if i == n_hours*3600//30:
        break
```

---

## CÉLULA 6 — Download features (original do notebook, sem alterações)

```python
!wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
!wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
```

---

## CÉLULA 7 — Carregar config (original do notebook, sem alterações)

```python
config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)
config
```

---

## CÉLULA 8 — Modificar config (ALTERADA)

```python
config["target_phrase"] = ["anfitrião"]
config["model_name"] = "anfitriao"
config["n_samples"] = 200        # ajustar para o total de amostras train
config["n_samples_val"] = 50     # ajustar para o total de amostras test
config["steps"] = 5000
config["target_accuracy"] = 0.6
config["target_recall"] = 0.25
config["background_paths"] = ['./audioset_16k', './fma']
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open('my_model.yaml', 'w') as file:
    documents = yaml.dump(config, file)
```

---

## CÉLULA 9 — Gerar clips (original do notebook, sem alterações)

> Vai pular a geração de positivos (já existem) e gerar os negativos adversariais.

```python
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips
```

---

## CÉLULA 10 — Gerar negativos com gTTS (NOVA)

> Rodar logo após a Célula 9.

```python
!pip install gtts pydub -q
!apt-get install -y ffmpeg -q

from gtts import gTTS
from pydub import AudioSegment
import os, io

negative_train_dir = "./my_custom_model/anfitriao/negative_train"
negative_test_dir  = "./my_custom_model/anfitriao/negative_test"
os.makedirs(negative_train_dir, exist_ok=True)
os.makedirs(negative_test_dir, exist_ok=True)

negatives = [
    "anfiteatro", "antipatia", "anterior", "antigo", "anfíbio",
    "amigo", "animado", "aniversário", "antes", "antena",
    "antagonista", "anfitriã", "antiquado", "antídoto", "ânfora",
    "anfora", "antifaz", "anti-herói", "anteparo", "animação",
]

n_train_needed = max(0, 200 - len(os.listdir(negative_train_dir)))
n_test_needed  = max(0, 50  - len(os.listdir(negative_test_dir)))
all_texts = (negatives * 20)[:(n_train_needed + n_test_needed)]

for i, text in enumerate(all_texts):
    out_dir = negative_train_dir if i < n_train_needed else negative_test_dir
    fpath = os.path.join(out_dir, f"neg_{i:04d}.wav")
    mp3_buf = io.BytesIO()
    gTTS(text=text, lang='pt', tld='com.br').write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    AudioSegment.from_mp3(mp3_buf).set_frame_rate(16000).set_channels(1).export(fpath, format='wav')
    if (i + 1) % 50 == 0:
        print(f"{i+1} gerados")

print(f"Train: {len(os.listdir(negative_train_dir))} | Test: {len(os.listdir(negative_test_dir))}")
```

---

## CÉLULA 11 — Limpar features antigas (NOVA)

> Garante que a Célula 12 vai regenerar tudo do zero.

```python
import glob
for f in glob.glob('./my_custom_model/anfitriao/*.npy'):
    os.remove(f)
    print(f"Removido: {f}")
print("Pronto.")
```

---

## CÉLULA 12 — Augmentar clips (original do notebook, sem alterações)

```python
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips
```

---

## CÉLULA 13 — Instalar onnxscript (NOVA — rodar antes de treinar)

```python
!pip install onnxscript -q
```

---

## CÉLULA 14 — Treinar modelo (original do notebook, sem alterações)

```python
!{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model
```

---

## CÉLULA 15 — Verificar e baixar modelo (NOVA)

```python
import os
onnx_path = f"./my_custom_model/{config['model_name']}.onnx"
if os.path.exists(onnx_path):
    print(f"Arquivo encontrado: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
    print("Baixe pelo painel de arquivos à esquerda.")
else:
    print("Arquivo não encontrado.")
```

---

## Notas

- Se o Colab resetar, rode as Células 1 e 2 novamente antes de continuar.
- O arquivo `.onnx` final vai para `companion/models/anfitriao.onnx`.
- Ajuste `n_samples` e `n_samples_val` na Célula 8 conforme o número real de amostras no zip.
