FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
WORKDIR /work
COPY --chown=65532:65532 coding_worker.py /work/coding_worker.py
USER 65532:65532
ENTRYPOINT ["python", "/work/coding_worker.py"]
