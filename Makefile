INSTALL_DIR ?= /data/scripts
BIN = rotatevideo

all: build

build:
	cargo build --release

debug:
	cargo build

test:
	cargo test --release
	cargo clippy --release -- -D warnings

install: build
	mkdir -p $(INSTALL_DIR)
	install -s -m 0755 target/release/$(BIN) $(INSTALL_DIR)/$(BIN)

# keep only target/release/$(BIN) and target/debug/$(BIN)
clean:
	[ -d target ] && find target -mindepth 1 -maxdepth 1 ! -name release ! -name debug -exec rm -rf {} + || true
	for d in target/release target/debug; do [ -d $$d ] && find $$d -mindepth 1 -maxdepth 1 ! -name $(BIN) -exec rm -rf {} + || true; done

distclean:
	cargo clean

.PHONY: all build debug test install clean distclean
