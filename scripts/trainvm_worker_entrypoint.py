"""Fixed code artifact for authority-launched TrainVM Python workers."""

from rwkv_lab.trainvm_adapters.entrypoint import main

if __name__ == "__main__":
    raise SystemExit(main())
