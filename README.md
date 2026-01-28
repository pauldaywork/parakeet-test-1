

# Tasks
[x] Run model
[x] Cache models
[x] Save output to file
[ ] wtf is TDT, CTC and RNNT?
      - CTC is non-autoregressive model: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#conformer-ctc
[ ] Check what other feature Riva supports
[ ] Diarization with preset speakers via audio samples
      - https://github.com/nvidia-riva/tutorials/blob/main/asr-speaker-diarization.ipynb
      - https://docs.nvidia.com/nim/riva/asr/latest/support-matrix.html#speech-recognition-with-vad-and-speaker-diarization
        - NIM_TAGS_SELECTOR="name=parakeet-0-6b-ctc-en-us,mode=ofl,diarizer=sortformer,vad=silero" tags to use diarization
          - requires 12g vram coz batch size = 1024
          - without diarizer and batch size = 1 it takes up 3g "name=parakeet-0-6b-ctc-en-us,bs=1,mode=ofl,diarizer=disabled,vad=default"
      - figure out wtf this is and if it will help https://docs.nvidia.com/nim/riva/asr/latest/pipeline-configuration.html
[ ] Batch process audio files
      - https://docs.nvidia.com/nim/riva/asr/latest/performance.html
      - https://docs.nvidia.com/nim/riva/asr/latest/deploy-helm.html (for more than 2 gpu's)
[ ] Keep model loaded to save time

# Run Client
```
uv venv --python 3.12 
source .venv/bin/activate
uv pip install -r requirements.txt 
uv run test.py
```

# Run Model API (Nvidia NIM)
```
export NGC_API_KEY=nvapi-????
# If using bash
echo "export NGC_API_KEY=<value>" >> ~/.bashrc

# If using zsh
echo "export NGC_API_KEY=<value>" >> ~/.zshrc

export LOCAL_NIM_CACHE=~/.cache/nim

mkdir -p $LOCAL_NIM_CACHE
chmod 777 $LOCAL_NIM_CACHE

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker run -it --rm \
   --runtime=nvidia \
   --gpus '"device=0"' \
   --shm-size=8GB \
   -e NGC_API_KEY \
   -e NIM_HTTP_API_PORT=9000 \
   -e NIM_GRPC_API_PORT=50051 \
   -p 9000:9000 \
   -p 50051:50051 \
   -e NIM_TAGS_SELECTOR=name=parakeet-0-6b-ctc-en-us,mode=ofl,bs=1 \
   -v ~/.cache/nim:/opt/nim/.cache \
   nvcr.io/nim/nvidia/parakeet-0-6b-ctc-en-us:latest

# check api is working
curl -X 'GET' 'http://localhost:9000/v1/health/ready'

# test a file out
curl -s http://0.0.0.0:9000/v1/audio/transcriptions -F language=en \
   -F file="@en-US_sample.wav"

conda create -n cuda128_py312_para python=3.12
conda activate cuda128_py312_para
conda install cuda=12.8 -c nvidia/label/cuda-12.8.1 
# Check the installed CUDA version
nvcc --version  

pip install nvidia-riva-client IPython
python3 test.py
```

# Run with diarization
```
docker run -it --rm \
   --runtime=nvidia \
   --gpus '"device=0"' \
   --shm-size=8GB \
   -e NGC_API_KEY \
   -e NIM_HTTP_API_PORT=9000 \
   -e NIM_GRPC_API_PORT=50051 \
   -p 9000:9000 \
   -p 50051:50051 \
   -e NIM_TAGS_SELECTOR=name=parakeet-0-6b-ctc-en-us,mode=ofl,diarizer=sortformer,vad=silero \
   -v ~/.cache/nim:/opt/nim/.cache \
   nvcr.io/nim/nvidia/parakeet-0-6b-ctc-en-us:latest
```

# Links
https://github.com/nvidia-riva
https://github.com/nvidia-riva/tutorials
https://huggingface.co/collections/nvidia/parakeet
https://github.com/nvidia-riva/python-clients/tree/main
https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-performance.html
https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-pipeline-configuration.html
https://docs.nvidia.com/nim/riva/asr/latest/configuration.html
https://docs.nvidia.com/nim/riva/asr/latest/getting-started-wsl.html
https://docs.nvidia.com/nim/riva/asr/latest/getting-started.html
Response object: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/protos.html#_CPPv426StreamingRecognitionResult (This response object is dog shit, fuck nvidia) https://docs.nvidia.com/deeplearning/riva/archives/160-b/user-guide/docs/protobuf-api/jarvis_asr.proto.html#_CPPv48WordInfo
Only useful way of parsing response: https://www.google.com/search?q=process+nvidia+riva+RecognizeResponse
audio chunk iterator: https://github.com/nvidia-riva/python-clients/blob/main/riva/client/asr.py#L49
