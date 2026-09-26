FROM node@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c
WORKDIR /work
COPY --chown=65532:65532 coding_worker.mjs /work/coding_worker.mjs
USER 65532:65532
ENTRYPOINT ["node", "/work/coding_worker.mjs"]
