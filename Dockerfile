FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY lforla_eval ./lforla_eval

RUN python -m pip install --no-cache-dir --prefix=/opt/lforla-eval . \
    && find /opt/lforla-eval -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/lforla-eval/bin:$PATH" \
    PYTHONPATH="/opt/lforla-eval/lib/python3.12/site-packages" \
    LFORLA_API_URL="https://lforla.org/api/v1"

RUN addgroup --system --gid 1000 lforla \
    && adduser --system --uid 1000 --gid 1000 --home /home/lforla --shell /usr/sbin/nologin lforla

COPY --from=builder /opt/lforla-eval /opt/lforla-eval

WORKDIR /data
RUN mkdir -p /data/reports /home/lforla/.config/lforla \
    && chown -R lforla:lforla /data /home/lforla

USER lforla:lforla

VOLUME ["/data"]

ENTRYPOINT ["lforla-eval"]
CMD ["--help"]
