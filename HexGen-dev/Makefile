.PHONY: subsystem
subsystem: go-install
	export PATH=${PATH}:/usr/local/go/bin && cd ./third_party/ocf/src/ocf-core && git init && make build

.PHONY: go-install
go-install:
	ls /usr/local/go/bin || wget -c https://dl.google.com/go/go1.20.linux-amd64.tar.gz -O - | sudo tar -xz -C /usr/local

.PHONY: requirements
requirements:
	pip install -r requirements.txt

.PHONY: flash-attn
flash-attn:
	pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install flash-attn==2.0.8
	-git clone https://github.com/Dao-AILab/flash-attention.git
	cd flash-attention && git submodule update --init csrc/cutlass && cd csrc/fused_dense_lib && pip install . \
	&& cd ../xentropy && pip install . && cd ../rotary && pip install . && cd ../layer_norm && pip install .

.PHONY: hexgen
hexgen: subsystem requirements flash-attn

.PHONY: hexgen-head
hexgen: subsystem requirements

.PHONY: clean
clean:
	-rm edit $(objects)
